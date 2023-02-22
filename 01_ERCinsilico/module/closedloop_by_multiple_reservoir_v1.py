####
#### Multi-ERC closed-loop
####

# Load modules
import numpy as np
import pandas as pd
import time
from scipy import linalg

def SimplexProjection(vector, select_lib, nn_number, tp = 1): #---------------------------#
    # Calculate distances and removed the last data (it has no future values)
    present_lib = select_lib[0:select_lib.shape[0] - tp]
    future_lib = select_lib[tp:select_lib.shape[0]]
    
    square = np.sum((present_lib - vector)**2, axis = 1)
    distances = np.sqrt(np.array(square, dtype = "float"))
    
    # Identifying IDs that include NaN
    not_nan_dist_id =  np.where(np.isnan(distances) == False)
    not_nan_future_id =  np.where(np.sum(np.isnan(future_lib), axis = 1) == 0)
    not_nan_id = np.intersect1d(not_nan_dist_id, not_nan_future_id)
    #not_nan_id = np.intersect1d(not_nan_dist_id, not_nan_future_id, not_zero_lib_id)
    
    # Identifying nearest neighbors and thier future vectors
    nn_distances_cand = distances[not_nan_id]
    future_lib_cand = future_lib[not_nan_id]
    nn_distances = nn_distances_cand[np.argsort(nn_distances_cand)][range(nn_number)]
    nn_future_vectors = future_lib_cand[np.argsort(nn_distances_cand)][range(nn_number)]
    
    # Check wheter nn_future_vectors include NaN
    min_distance = np.nanmin(nn_distances)
    if min_distance == 0:
        min_distance = 1e-6
        
    weights = np.exp(-nn_distances/min_distance)
    total_weight = sum(weights)
    pred_vector = (weights / total_weight) @ nn_future_vectors
    
    return np.array(pred_vector, dtype = "float")
#--------------------------------------------------------------------------------#

# Define class "SimplexReservoir"
class MultinetworkSimplexReservoir(): #=========================================================================#
    # Step 1: Initialize class "SimplexReservoir"
    def __init__(self, network_name):
        self.network = network_name
    
    # Step.2: Learn weights by softmax regression
    def learn_model(self, combined_reservoir_state, train_true, washout_fraction = 0, ridge_lambda = 0.05, const_input = 0.2):
        start_time_learnmodel = time.time()
        self.const_input = const_input
        self.combined_reservoir_state = combined_reservoir_state
        self.ridge_lambda = ridge_lambda
        self.train_true = train_true
        
        self.washout = int(washout_fraction * self.combined_reservoir_state.shape[0])
        washed_nrow = self.combined_reservoir_state[self.washout:-1,].shape[0]
        const_input_mat = np.repeat(self.const_input, washed_nrow).reshape(washed_nrow,1)
        combined_state_washed = np.hstack([self.combined_reservoir_state[self.washout:-1,], const_input_mat]) # Remove the last row
        
        # Ridge Regression
        E_lambda = np.identity(combined_state_washed.shape[1]) * self.ridge_lambda
        inv_x = np.linalg.inv(combined_state_washed.T @ combined_state_washed + E_lambda)
        # update weights of output layer
        self.Wout = (inv_x @ combined_state_washed.T) @ np.array(self.train_true[(1+self.washout):])
        self.train_predicted = combined_state_washed @ self.Wout
        self.train_pred = np.corrcoef([self.train_true[(1+self.washout):]], [self.train_predicted])[1,0]
        self.train_rmse = np.sqrt(np.mean((self.train_predicted - np.array(self.train_true[(1+self.washout):]))**2))
        self.train_nmse = sum((np.array(self.train_true[(1+self.washout):]) - self.train_predicted)**2)/sum(np.array(self.train_true[(1+self.washout):])**2)
        self.learnmodel_time = time.time() - start_time_learnmodel
    
    # Step.3: Predict test data
    def predict(self, test_true, reservoir_obj):
        if type(reservoir_obj) is list:
            self.n_reservoir = len(reservoir_obj)
        else:
            self.n_reservoir = len([reservoir_obj])
        self.test_true = test_true
        r = reservoir_obj
        
        # Collect num_nodes
        if self.n_reservoir == 1:
            n_each_nodes = r.num_nodes
            self.num_nodes = n_each_nodes
        else:
            r_df = r[0].result_summary_df
            for i in range(1,len(r)): r_df = r_df.append(r[i].result_summary_df)
            n_each_nodes = np.array(r_df["num_nodes"].astype("int"))
            self.num_nodes = n_each_nodes.sum()

        record_reservoir_test_nrow = int(self.test_true.shape[0] + 1)
        self.combined_test_reservoir_nodes = np.zeros((record_reservoir_test_nrow, self.num_nodes))
        self.combined_test_reservoir_nodes[0,:] = self.combined_reservoir_state[-1]
        
        for net_i in np.arange(0, self.n_reservoir):
            if self.n_reservoir == 1:
                ni1 = n_each_nodes
                ni2 = 0
                I = np.identity(ni1)
            else:
                ni1 = n_each_nodes[net_i]
                ni2 = n_each_nodes[:net_i].sum()
                I = np.identity(ni1)
            
            for data_i, test_back in enumerate(self.test_true):
                x0 = self.combined_test_reservoir_nodes[data_i, ni2:(ni1+ni2)]
                u1 = np.array([self.const_input])
                concat_reservoir_nodes = np.hstack([self.combined_test_reservoir_nodes[data_i,:], self.const_input])
                y0 = np.array([concat_reservoir_nodes @ self.Wout])
                if self.n_reservoir == 1:
                    x1 = (1 - r.C1 * r.a1) * x0 + r.C1 * ((SimplexProjection(u1 @ r.Win + x0, r.db, r.n_nn) + y0 @ r.Wback + r.bias))
                else:
                    x1 = (1 - r[net_i].C1 * r[net_i].a1) * x0 + r[net_i].C1 * ((SimplexProjection(u1 @ r[net_i].Win + x0, r[net_i].db, r[net_i].n_nn) + y0 @ r[net_i].Wback + r[net_i].bias))
                self.combined_test_reservoir_nodes[data_i + 1, ni2:(ni1+ni2)] = x1
        
        washed_nrow = self.combined_test_reservoir_nodes.shape[0]
        const_input_mat = np.repeat(self.const_input, washed_nrow).reshape(washed_nrow,1)
        concat_test_reservoir_nodes = np.hstack([self.combined_test_reservoir_nodes, const_input_mat]) # Remove the last row
        self.test_predicted = concat_test_reservoir_nodes[:-1,:] @ self.Wout
        self.test_pred = np.corrcoef([self.test_true], [self.test_predicted])[1,0]
        self.test_rmse = np.sqrt(np.mean((np.array(self.test_true) - self.test_predicted)**2))
        self.test_nmse = sum((np.array(self.test_true) - self.test_predicted)**2)/sum(np.array(self.test_true)**2)
        
    # Step. 4: Summarize stats
    def summarize_stat(self):
        # Summary statistics
        result_summary = np.array([self.network, round(self.train_pred, 4), round(self.test_pred, 4),
                                 round(self.train_rmse, 4), round(self.test_rmse, 4),
                                 round(self.train_nmse, 7), round(self.test_nmse, 7),
                                 self.combined_reservoir_state.shape[1], round(self.learnmodel_time, 2)])
        self.result_summary_df = pd.DataFrame(result_summary.reshape(1,len(result_summary)))
        self.result_summary_df = self.result_summary_df.rename(columns = {0:"network_name",
                                                                          1:"train_pred", 2:"test_pred",
                                                                          3:"RMSE_train", 4:"RMSE_test",
                                                                          5:"NMSE_train", 6:"NMSE_test",
                                                                          7:"total_nodes", 8:"learning_time"})
#====================================================================================================#


