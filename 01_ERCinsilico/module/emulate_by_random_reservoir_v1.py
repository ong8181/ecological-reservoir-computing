####
#### ESN emulate
####

# Load modules
import numpy as np
import pandas as pd
import time
from scipy import linalg

# Define functions
### Calculate the next state (reservoir dynamics)
def calculate_next_state(input_data, reservoir_nodes_t0, Win, W, leak, bias = 0): #------------------------#
    # Define identity matrixa
    I = np.identity(reservoir_nodes_t0.shape[0])
    # Set the current state
    t0 = np.array(reservoir_nodes_t0)
    t1 = leak @ t0 + np.tanh([input_data] @ Win + (I - leak) @ t0 @ W + bias)
    return(t1)
#---------------------------------------------------------------------------#


class RandomReservoir(): #=========================================================================#
    # Step 1: Initialize class "RandomReservoir"
    def __init__(self, network_name = "Random_Network"):
        self.network = network_name
    
    # Step 2: Prepare target data
    def prepare_target_data(self, target_ts, target_var, true_ts, true_var, test_fraction):
        self.target_data  = target_ts[target_var]
        self.target_var = target_var
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
        self.test_data = std_data[train_size:(data_size - 1)].reset_index()[self.target_var]
        self.train_true = std_true_data[0:train_size].reset_index()[self.true_var]
        self.test_true = std_true_data[train_size:(data_size - 1)].reset_index()[self.true_var]
    
    # Step 3: Initialize reservoir
    def initialize_reservoir(self, num_reservoir_nodes, alpha, w_sparsity = 0.9, w_in_sparsity = 0.9,
                             w_in_strength = 0.6, leak_rate = 0, num_input_nodes = 1, num_output_nodes = 1,
                             Win_seed = 1234, W_seed = 1235):
        # Set parameters
        self.num_reservoir_nodes = num_reservoir_nodes
        self.alpha = alpha
        self.w_sparsity = w_sparsity
        self.w_in_sparsity = w_in_sparsity
        self.w_in_strength = w_in_strength
        self.leak_rate = leak_rate
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.Win_seed = Win_seed
        self.W_seed = W_seed
        
        # Initialize W_in
        self.R = np.identity(self.num_reservoir_nodes) * self.leak_rate
        np.random.seed(self.Win_seed); W_in0 = np.random.uniform(-1, 1, (self.num_input_nodes * self.num_reservoir_nodes, 1))
        np.random.seed(self.Win_seed); rand_id = np.random.choice(W_in0.shape[0], int(self.num_input_nodes * self.num_reservoir_nodes * self.w_in_sparsity), replace = False)
        W_in0[rand_id] = 0; self.W_in = W_in0.reshape(self.num_input_nodes, self.num_reservoir_nodes) * self.w_in_strength
        
        # Initialize W
        np.random.seed(self.W_seed); W0 = np.random.uniform(-1, 1, (self.num_reservoir_nodes * self.num_reservoir_nodes, 1))
        np.random.seed(self.W_seed); rand_id = np.random.choice(W0.shape[0], int(self.num_reservoir_nodes * self.num_reservoir_nodes * self.w_sparsity), replace = False)
        W0[rand_id] = 0; W1 = W0.reshape(self.num_reservoir_nodes, self.num_reservoir_nodes)
        ## Normalize W0 using spectral radius
        W2 = W1 / abs(np.linalg.eig(W1)[0]).max()
        ## Scale W1 to W using alpha
        self.W = W2 * self.alpha
    
    # Step 4: Compute reservoir states
    def compute_reservoir_state(self):
        start_time_training = time.time()
        record_reservoir_train_nrow = int(self.train_data.shape[0] + 1) # size of the training data
        self.record_reservoir_nodes = np.zeros((record_reservoir_train_nrow, self.num_reservoir_nodes))
        
        # Calculate the next state
        for data_i, input_train in enumerate(self.train_data):
            x_n1 = calculate_next_state(input_train, self.record_reservoir_nodes[data_i,:],
                                        self.W_in, self.W, self.R)
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
        self.train_rmse = np.sqrt(np.mean((self.train_predicted - np.array(self.train_true[self.washout:]))**2))
        self.train_nmse = sum((np.array(self.train_true[self.washout:]) - self.train_predicted)**2)/sum(np.array(self.train_true[self.washout:])**2)
        self.learnmodel_time = time.time() - start_time_learnmodel
    
    def predict(self):
        start_time_testing = time.time()
        # Step.5: Exploitation
        record_reservoir_test_nrow = int(self.test_data.shape[0] + 1)
        self.test_reservoir_nodes = np.zeros((record_reservoir_test_nrow, self.num_reservoir_nodes))
        self.test_reservoir_nodes[0,:] = self.record_reservoir_nodes[-1]
        
        for data_i, input_test in enumerate(self.test_data):
            x_n1 = calculate_next_state(input_test, self.test_reservoir_nodes[data_i,:],
                                        self.W_in, self.W, self.R)
            self.test_reservoir_nodes[data_i + 1,:] = x_n1
        
        self.test_reservoir_nodes = self.test_reservoir_nodes[1:,]
        self.test_predicted = self.test_reservoir_nodes @ self.W_out
        self.test_pred = np.corrcoef([self.test_true], [self.test_predicted])[1,0]
        self.test_rmse = np.sqrt(np.mean((np.array(self.test_true) - self.test_predicted)**2))
        self.test_nmse = sum((np.array(self.test_true) - self.test_predicted)**2)/sum(np.array(self.test_true)**2)
        self.testing_time = time.time() - start_time_testing
    
    def summarize_stat(self):
        # Summary statistics
        result_summary = np.array([self.network, round(self.train_pred, 4), round(self.test_pred, 4),
                                 round(self.train_rmse, 4), round(self.test_rmse, 4),
                                 round(self.train_nmse, 7), round(self.test_nmse, 7),
                                 self.num_reservoir_nodes, self.w_in_strength, self.w_in_sparsity,
                                 self.alpha, self.w_sparsity, round(self.leak_rate, 2), self.target_data.shape[0],
                                 self.test_fraction, self.washout,
                                 round(self.training_time, 2), round(self.learnmodel_time, 2), round(self.testing_time, 2)])
        self.result_summary_df = pd.DataFrame(result_summary.reshape(1,19))
        self.result_summary_df = self.result_summary_df.rename(columns = {0:"network",
                                                          1:"train_pred", 2:"test_pred",
                                                          3:"RMSE_train", 4:"RMSE_test",
                                                          5:"NMSE_train", 6:"NMSE_test",
                                                          7:"num_nodes", 8:"Win_strength", 9:"Win_sparsity",
                                                          10:"alpha", 11:"W_sparsity", 12:"leak_rate", 13:"data_size",
                                                          14:"test_fraction", 15:"washout_data",
                                                          16:"training_time", 17:"learnmodel_time", 18:"testing_time"})
#=========================================================================#
