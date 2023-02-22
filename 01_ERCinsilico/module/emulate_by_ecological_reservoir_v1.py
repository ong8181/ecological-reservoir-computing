####
#### ERC emulate
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
    def __init__(self, reservoir_var_index, reservoir_ts_data, reservoir_db_name):
        self.reservoir_var = reservoir_var_index
        self.reservoir_ts = reservoir_ts_data
        self.reservoir_db = reservoir_db_name
    
    # Step 2: Select and compile library data
    def compile_reservoir_data(self, num_reservoir_nodes):
        self.num_nodes = num_reservoir_nodes
        ts1 = self.reservoir_ts[self.reservoir_var]
        # Standardize data (from 0 to 1) if time series is not "zero"
        if self.reservoir_var != 'zeros':
            ts1 = (ts1 - ts1.min()) / (ts1.max() - ts1.min())
        # Time-delay embedding to reconstruct state-space
        self.db = embed(ts1, E = self.num_nodes, tp = 1)
    
    # Step 2: Prepare target data
    def prepare_target_data(self, target_ts, target_var, target_db_name, true_ts, true_var, test_fraction = 0.2):
        self.target_data  = target_ts[target_var]
        self.target_var = target_var
        self.target_db = target_db_name
        self.true_data = true_ts[true_var]
        self.true_var = true_var
        self.test_fraction = test_fraction
        # Standardize data
        std_data = (self.target_data - self.target_data.min()) / (self.target_data.max() - self.target_data.min())
        std_true_data = (self.true_data - self.true_data.min()) / (self.true_data.max() - self.true_data.min())
        
        # Split data into training data set and test data set
        data_size = self.target_data.shape[0]
        test_size = int(round(data_size*self.test_fraction))
        train_size = int(data_size - test_size)
        self.train_data = std_data[0:train_size].reset_index()[self.target_var]
        self.test_data = std_data[train_size:].reset_index()[self.target_var]
        self.train_true = std_true_data[0:train_size].reset_index()[self.true_var]
        self.test_true = std_true_data[train_size:].reset_index()[self.true_var]
    
    # Step 3: Initialize reservoir
    def initialize_reservoir(self, w_in_strength = 0.1, w_in_sparsity = 0.2, w_in_seed = 1234, n_nn = None,
                             leak_rate = 0, num_input_nodes = 1, num_output_nodes = 1):
        # Set primary parameters
        self.train_data_size = self.train_data.shape[0]
        self.w_in_strength = w_in_strength
        self.w_in_sparsity = w_in_sparsity
        self.leak_rate = leak_rate
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        if n_nn == None:
            self.n_nn = self.num_nodes + 1
        else:
            self.n_nn = n_nn
        
        # Set secondary parameters
        self.I = np.identity(self.num_nodes)
        self.R = self.I * self.leak_rate
        np.random.seed(w_in_seed); W_in0 = np.random.uniform(-1, 1, (self.num_input_nodes * self.num_nodes, 1))
        np.random.seed(w_in_seed); rand_id = np.random.choice(W_in0.shape[0], int(self.num_input_nodes * self.num_nodes * self.w_in_sparsity), replace = False)
        W_in0[rand_id] = 0; self.W_in = W_in0.reshape(self.num_input_nodes, self.num_nodes) * self.w_in_strength
    
    # Step 4: Compute reservoir states
    def compute_reservoir_state(self, initial_method = "zero", manual_index = None):
        start_time_training = time.time()
        record_reservoir_train_nrow = int(self.train_data.shape[0] + 1) # size of the training data
        self.record_reservoir_nodes = np.zeros((record_reservoir_train_nrow, self.num_nodes))
        
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
        for data_i, input_train in enumerate(self.train_data):
            x_n1 = simplex_next_state(self.record_reservoir_nodes[data_i,:], # Previous vector (t0)
                                      [input_train] @ self.W_in, # Perturbation (input)
                                      self.n_nn, self.db, self.R)
            self.record_reservoir_nodes[data_i + 1,:] = x_n1
        
        self.record_reservoir_nodes = self.record_reservoir_nodes[1:,]
        self.training_time = time.time() - start_time_training
    
    # Step 5: Learn model (ridge regression)
    def learn_model(self, washout_fraction = 0, ridge_lambda = 0.05):
        start_time_learnmodel = time.time()
        self.ridge_lambda = ridge_lambda
        self.washout = int(washout_fraction * self.train_data.shape[0])
        reservoir_nodes_washed = self.record_reservoir_nodes[self.washout:,]
        # Ridge Regression
        E_lambda = np.identity(reservoir_nodes_washed.shape[1]) * self.ridge_lambda
        inv_x = np.linalg.inv(reservoir_nodes_washed.T @ reservoir_nodes_washed + E_lambda)
        # update weights of output layer
        self.W_out = (inv_x @ reservoir_nodes_washed.T) @ np.array(self.train_true[self.washout:])
        self.train_predicted = reservoir_nodes_washed @ self.W_out
        self.train_pred = np.corrcoef([self.train_true[self.washout:]], [self.train_predicted])[1,0]
        self.train_rmse = np.sqrt(np.mean((np.array(self.train_true[self.washout:] - self.train_predicted))**2))
        self.train_nmse = sum((np.array(self.train_true[self.washout:]) - self.train_predicted)**2)/sum(np.array(self.train_true[self.washout:])**2)
        self.learnmodel_time = time.time() - start_time_learnmodel
    
    def predict(self):
        start_time_testing = time.time()
        # Step.5: Exploitation
        record_reservoir_test_nrow = int(self.test_data.shape[0] + 1)
        self.test_reservoir_nodes = np.zeros((record_reservoir_test_nrow, self.num_nodes))
        self.test_reservoir_nodes[0,:] = self.record_reservoir_nodes[-1]
        
        for data_i, input_test in enumerate(self.test_data):
            x_n1 = simplex_next_state(self.test_reservoir_nodes[data_i,:],
                                      [input_test] @ self.W_in,
                                      self.n_nn, self.db, self.R)
            self.test_reservoir_nodes[data_i + 1,:] = x_n1
        
        self.test_reservoir_nodes = self.test_reservoir_nodes[1:,]
        self.test_predicted = self.test_reservoir_nodes @ self.W_out
        self.test_pred = np.corrcoef([self.test_true], [self.test_predicted])[1,0]
        self.test_rmse = np.sqrt(np.mean((self.test_predicted - np.array(self.test_true))**2))
        self.test_nmse = sum((np.array(self.test_true) - self.test_predicted)**2)/sum(np.array(self.test_true)**2)
        self.testing_time = time.time() - start_time_testing
    
    # Step. 7: Summarize stats
    def summarize_stat(self):
        # Summary statistics
        result_summary = np.array([self.reservoir_var, self.reservoir_db, self.target_var, self.target_db,
                                 round(self.train_pred, 4), round(self.test_pred, 4),
                                 round(self.train_rmse, 4), round(self.test_rmse, 4),
                                 round(self.train_nmse, 7), round(self.test_nmse, 7),
                                 self.num_nodes, self.w_in_strength, self.w_in_sparsity,
                                 self.n_nn, round(self.leak_rate, 2), self.train_data_size,
                                 self.test_fraction, self.washout,
                                 round(self.training_time, 2), round(self.learnmodel_time, 2), round(self.testing_time, 2)])
        self.result_summary_df = pd.DataFrame(result_summary.reshape(1,21))
        self.result_summary_df = self.result_summary_df.rename(columns = {0:"reservoir_var",
                                                          1:"reservoir_db_name", 2:"target_var", 3:"target_db_name",
                                                          4:"train_pred", 5:"test_pred",
                                                          6:"RMSE_train", 7:"RMSE_test",
                                                          8:"NMSE_train", 9:"NMSE_test",
                                                          10:"num_nodes", 11:"Win_strength",
                                                          12:"Win_sparsity", 13:"n_neighbors",
                                                          14:"leak_rate", 15:"subset_size",
                                                          16:"test_fraction", 17:"washout_data",
                                                          18:"training_time", 19:"learnmodel_time", 20:"testing_time"})
#====================================================================================================#


