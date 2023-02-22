####
#### ERC closed-loop
####

# Load modules
import numpy as np
import pandas as pd
import time
from scipy import linalg

# Define functions
# Simplex projection
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
    
    # Identifying nearest neighbors and thier future vectors
    nn_distances_cand = distances[not_nan_id]
    future_lib_cand = future_lib[not_nan_id]
    nn_distances = nn_distances_cand[np.argsort(nn_distances_cand)][range(nn_number)]
    nn_future_vectors = future_lib_cand[np.argsort(nn_distances_cand)][range(nn_number)]
    
    # Identify neighbor vectors
    #present_lib_cand = present_lib[not_nan_id]
    #nn_neighbor_vectors = present_lib_cand[np.argsort(nn_distances_cand)][range(nn_number)]
    
    # Check wheter nn_future_vectors include NaN
    min_distance = np.nanmin(nn_distances)
    if min_distance == 0:
        min_distance = 1e-6
        
    weights = np.exp(-nn_distances/min_distance)
    total_weight = sum(weights)
    pred_vector = (weights / total_weight) @ nn_future_vectors
    
    return np.array(pred_vector, dtype = "float")
#--------------------------------------------------------------------------------#

# Calculate the next state (reservoir dynamics)
def simplex_next_state(current_vector, input_vector, n_neighbors, library_ts, leak, tp = 1, bias = 0): #------------------------#
    # Define identity matrix
    I = np.identity(input_vector.shape[0])
    # Set the current state
    t0 = np.array(current_vector)
    t1 = leak @ t0 + (I - leak) @ ((SimplexProjection(input_vector + t0, library_ts, n_neighbors, tp = tp) + bias))
    return(t1)
#---------------------------------------------------------------------------#

### Embedding function ---------------------------------------------------------------------------#
def embed(ts_data, E = 2, tp = 1):
    ts1 = ts_data
    # Time-delay embedding to reconstruct state-space
    for i in range(E-1):
        nan_ts = np.repeat(np.nan, i*tp + 1)
        if i == 0:
            delay_ts = pd.concat([pd.Series(nan_ts), ts1.iloc[0:(-i*tp-1),]], axis = 0)
        else:
            delay_ts = pd.concat([pd.Series(nan_ts), ts1.iloc[0:(-i*tp-1),0]], axis = 0)
        delay_ts.index = pd.Index(range(ts1.shape[0]))
        ts1 = pd.concat([ts1, delay_ts], axis = 1)
    # Remove all rows containing any NaN
    ts1.columns = pd.Index(range(ts1.shape[1]))
    return ts1.values
#---------------------------------------------------------------------------#

# Define class "SimplexReservoir"
class SimplexReservoir(): #=========================================================================#
    # Step 1: Initialize class "SimplexReservoir"
    def __init__(self, task_name, reservoir_var_index, reservoir_ts_data, reservoir_db_name):
        self.task_name = task_name
        self.reservoir_var = reservoir_var_index
        self.reservoir_ts = reservoir_ts_data
        self.reservoir_db = reservoir_db_name
    
    # Step 2: Select and compile library data
    def compile_reservoir_data(self, num_reservoir_nodes = 10):
        self.num_nodes = num_reservoir_nodes
        ts1 = self.reservoir_ts[self.reservoir_var]
        # Standardize data (from 0 to 1) if time series is not "zero"
        if self.reservoir_var != 'zeros':
            ts1 = (ts1 - ts1.min()) / (ts1.max() - ts1.min())
        # Time-delay embedding to reconstruct state-space
        self.db = embed(ts1, E = self.num_nodes, tp = 1)
    
    # Step 2: Prepare target data
    def prepare_data(self, train_data, train_true, test_data, test_true,
                     train_var = "", test_var = ""):
        self.train_var = train_var
        self.test_var = test_var
        
        # Split data into training data set and test data set
        self.train_data = np.array(train_data)
        self.test_data = np.array(test_data)
        self.train_true = np.array(train_true)
        self.test_true = np.array(test_true)
        self.train_data_size = len(train_data)
        self.test_data_size = len(test_data)
        self.test_fraction = round(len(test_data)/(len(train_data) + len(test_data)), 3)
    
    # Step 3: Initialize reservoir
    def initialize_reservoir(self, w_in_strength = 1, w_in_sparsity = 0, w_back_strength = 1, w_back_sparsity = 0,
                             n_nn = None, num_input_nodes = 1, num_output_nodes = 1,
                             Win_seed = 1234, Wback_seed = 1236):
        # Set primary parameters
        self.train_data_size = self.train_true.shape[0]
        self.w_in_strength = w_in_strength
        self.w_in_sparsity = w_in_sparsity
        self.w_back_strength = w_back_strength
        self.w_back_sparsity = w_back_sparsity
        self.Win_seed = Win_seed
        self.Wback_seed = Wback_seed
        #self.leak_rate = leak_rate
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        if n_nn == None:
            self.n_nn = self.num_nodes + 1
        else:
            self.n_nn = n_nn
        
        # Set secondary parameters
        np.random.seed(self.Win_seed); Win0 = np.random.uniform(-1, 1, (self.num_input_nodes * self.num_nodes, 1))
        np.random.seed(self.Win_seed); rand_id = np.random.choice(Win0.shape[0], int(self.num_input_nodes * self.num_nodes * self.w_in_sparsity), replace = False)
        Win0[rand_id] = 0; self.Win = Win0.reshape(self.num_input_nodes, self.num_nodes) * self.w_in_strength
        
        # Initialize W_back
        np.random.seed(self.Wback_seed); Wback0 = np.random.uniform(-1, 1, (self.num_nodes * self.num_output_nodes, 1))
        np.random.seed(self.Wback_seed); rand_id = np.random.choice(Wback0.shape[0], int(self.num_nodes * self.num_output_nodes * self.w_back_sparsity), replace = False)
        Wback0[rand_id] = 0; self.Wback = Wback0.reshape(self.num_output_nodes, self.num_nodes) * self.w_back_strength
    
    # Step 4: Compute reservoir states
    def compute_reservoir_state(self, const_input = 0.2, C1 = 0.44, a1 = 0.9, bias = 0,
                                initial_method = "zero", manual_index = None):
        start_time_training = time.time()
        record_reservoir_train_nrow = int(self.train_true.shape[0] + 1) # size of the training data
        self.record_reservoir_nodes = np.zeros((record_reservoir_train_nrow, self.num_nodes))
        I = np.identity(self.record_reservoir_nodes.shape[1])
        self.const_input = const_input
        self.C1 = C1
        self.a1 = a1
        self.leak_rate = (1-C1*a1)
        self.bias = bias
        
        # Set iniital state
        if initial_method == "zero":
            self.record_reservoir_nodes[0,:] = 0
        elif initial_method == "random":
            random_id = np.random.randint(self.num_nodes-1, self.db.shape[0])
            self.record_reservoir_nodes[0,:] = self.db[random_id]
        elif initial_method == "manual":
            self.record_reservoir_nodes[0,:] = self.db[manual_index]
        else:
            print("Error: Specify correct initial_method argument.")
            sys.exit(1)
        
        # Calculate the next state
        for data_i, train_back in enumerate(self.train_true):
            x0 = self.record_reservoir_nodes[data_i,:]
            u1 = np.array([self.const_input])
            y0 = np.array([train_back])
            x1 = (1 - self.C1 * self.a1) * x0 + self.C1 * ((SimplexProjection(u1 @ self.Win + x0, self.db, self.n_nn) + y0 @ self.Wback + self.bias))
            self.record_reservoir_nodes[data_i + 1,:] = x1
        
        self.record_reservoir_nodes = self.record_reservoir_nodes[1:,]
        self.training_time = time.time() - start_time_training
    
    # Step 5: Learn model (ridge regression)
    def learn_model(self, washout_fraction = 0, ridge_lambda = 0.05):
        start_time_learnmodel = time.time()
        self.ridge_lambda = ridge_lambda
        self.washout = int(washout_fraction * self.train_true.shape[0])
        washed_nrow = self.record_reservoir_nodes[self.washout:-1,].shape[0]
        const_input_mat = np.repeat(self.const_input, washed_nrow).reshape(washed_nrow,1)
        reservoir_nodes_washed = np.hstack([self.record_reservoir_nodes[self.washout:-1,], const_input_mat]) # Remove the last row
        # Ridge Regression
        E_lambda = np.identity(reservoir_nodes_washed.shape[1]) * self.ridge_lambda
        inv_x = np.linalg.inv(reservoir_nodes_washed.T @ reservoir_nodes_washed + E_lambda)
        # update weights of output layer
        self.Wout = (inv_x @ reservoir_nodes_washed.T) @ np.array(self.train_true[(1+self.washout):])
        self.train_predicted = reservoir_nodes_washed @ self.Wout
        self.train_pred = np.corrcoef([self.train_true[(1+self.washout):]], [self.train_predicted])[1,0]
        self.train_rmse = np.sqrt(np.mean((self.train_predicted - np.array(self.train_true[(1+self.washout):]))**2))
        self.train_nmse = sum((np.array(self.train_true[(1+self.washout):]) - self.train_predicted)**2)/sum(np.array(self.train_true[(1+self.washout):])**2)
        self.learnmodel_time = time.time() - start_time_learnmodel
    
    def predict(self):
        start_time_testing = time.time()
        # Step.5: Exploitation
        record_reservoir_test_nrow = int(self.test_true.shape[0] + 1)
        self.test_reservoir_nodes = np.zeros((record_reservoir_test_nrow, self.num_nodes))
        self.test_reservoir_nodes[0,:] = self.record_reservoir_nodes[-1]
        I = np.identity(self.record_reservoir_nodes.shape[1])
        
        for data_i, test_back in enumerate(self.test_true):
            # Set the current state
            x0 = self.test_reservoir_nodes[data_i,:]
            u1 = np.array([self.const_input])
            concat_reservoir_nodes = np.hstack([self.test_reservoir_nodes[data_i,:], self.const_input])
            y0 = np.array([concat_reservoir_nodes @ self.Wout])
            x1 = (1 - self.C1 * self.a1) * x0 + self.C1 * ((SimplexProjection(u1 @ self.Win + x0, self.db, self.n_nn) + y0 @ self.Wback + self.bias))
            self.test_reservoir_nodes[data_i + 1,:] = x1
        
        washed_nrow = self.test_reservoir_nodes.shape[0]
        const_input_mat = np.repeat(self.const_input, washed_nrow).reshape(washed_nrow,1)
        concat_test_reservoir_nodes = np.hstack([self.test_reservoir_nodes, const_input_mat]) # Remove the last row
        self.test_predicted = concat_test_reservoir_nodes[:-1,:] @ self.Wout
        self.test_pred = np.corrcoef([self.test_true], [self.test_predicted])[1,0]
        self.test_rmse = np.sqrt(np.mean((np.array(self.test_true) - self.test_predicted)**2))
        self.test_nmse = sum((np.array(self.test_true) - self.test_predicted)**2)/sum(np.array(self.test_true)**2)
        self.testing_time = time.time() - start_time_testing
    
    # Step. 7: Summarize stats
    def summarize_stat(self):
        # Summary statistics
        result_summary = np.array([self.reservoir_var, self.reservoir_db, self.train_var, self.test_var,
                                 round(self.train_pred, 4), round(self.test_pred, 4),
                                 round(self.train_rmse, 4), round(self.test_rmse, 4),
                                 round(self.train_nmse, 7), round(self.test_nmse, 7),
                                 self.num_nodes, self.w_in_strength, self.w_in_sparsity,
                                 self.w_back_strength, self.w_back_sparsity,
                                 self.n_nn, round(self.leak_rate, 2), self.train_data_size + self.test_data_size,
                                 self.test_fraction, self.washout,
                                 round(self.training_time, 2), round(self.learnmodel_time, 2), round(self.testing_time, 2)])
        self.result_summary_df = pd.DataFrame(result_summary.reshape(1,23))
        self.result_summary_df = self.result_summary_df.rename(columns = {0:"reservoir_var",
                                                          1:"reservoir_db_name", 2:"train_var", 3:"test_var",
                                                          4:"train_pred", 5:"test_pred",
                                                          6:"RMSE_train", 7:"RMSE_test",
                                                          8:"NMSE_train", 9:"NMSE_test",
                                                          10:"num_nodes", 11:"Win_strength",
                                                          12:"Win_sparsity", 13:"Wback_strength",
                                                          14:"Wback_sparsity", 15:"n_neighbors",
                                                          16:"leak_rate", 17:"total_data_size",
                                                          18:"test_fraction", 19:"washout_data",
                                                          20:"training_time", 21:"learnmodel_time", 22:"testing_time"})
#====================================================================================================#


