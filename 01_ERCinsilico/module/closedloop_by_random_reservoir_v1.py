####
#### ESN closed-loop
####

# Load modules
import numpy as np
import pandas as pd
import time
from scipy import linalg


class RandomReservoir(): #=========================================================================#
    # Step 1: Initialize class "RandomReservoir"
    def __init__(self, network_name = "Closed_loop_embedding"):
        self.network = network_name
    
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
    def initialize_reservoir(self, num_reservoir_nodes, alpha, w_sparsity = 0.1, w_in_sparsity = 0.1, w_back_sparsity = 0.1,
                             w_in_strength = 1, w_back_strength = 1, leak_rate = None, num_input_nodes = 1, num_output_nodes = 1,
                             Win_seed = 1234, W_seed = 1235, Wback_seed = 1236):
        # Set parameters
        self.num_reservoir_nodes = num_reservoir_nodes
        self.alpha = alpha
        self.w_sparsity = w_sparsity
        self.w_in_sparsity = w_in_sparsity
        self.w_back_sparsity = w_back_sparsity
        self.w_in_strength = w_in_strength
        self.w_back_strength = w_back_strength
        #self.leak_rate = leak_rate
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.Win_seed = Win_seed
        self.W_seed = W_seed
        self.Wback_seed = Wback_seed
        
        # Initialize W_in
        np.random.seed(self.Win_seed); Win0 = np.random.uniform(-1, 1, (self.num_input_nodes * self.num_reservoir_nodes, 1))
        np.random.seed(self.Win_seed); rand_id = np.random.choice(Win0.shape[0], int(self.num_input_nodes * self.num_reservoir_nodes * self.w_in_sparsity), replace = False)
        Win0[rand_id] = 0; self.Win = Win0.reshape(self.num_input_nodes, self.num_reservoir_nodes) * self.w_in_strength
        
        # Initialize W
        np.random.seed(self.W_seed); W0 = np.random.uniform(-1, 1, (self.num_reservoir_nodes * self.num_reservoir_nodes, 1))
        np.random.seed(self.W_seed); rand_id = np.random.choice(W0.shape[0], int(self.num_reservoir_nodes * self.num_reservoir_nodes * self.w_sparsity), replace = False)
        W0[rand_id] = 0; W1 = W0.reshape(self.num_reservoir_nodes, self.num_reservoir_nodes)
        ## Normalize W0 using spectral radius
        W2 = W1 / abs(np.linalg.eig(W1)[0]).max()
        ## Scale W1 to W using alpha
        self.W = W2 * self.alpha
        
        # Initialize W_back
        np.random.seed(self.Wback_seed); Wback0 = np.random.uniform(-1, 1, (self.num_reservoir_nodes * self.num_output_nodes, 1))
        np.random.seed(self.Wback_seed); rand_id = np.random.choice(Wback0.shape[0], int(self.num_reservoir_nodes * self.num_output_nodes * self.w_back_sparsity), replace = False)
        Wback0[rand_id] = 0; self.Wback = Wback0.reshape(self.num_output_nodes, self.num_reservoir_nodes) * self.w_back_strength
    
    # Step 4: Compute reservoir states
    def compute_reservoir_state(self, const_input = 0.2, C1 = 0.44, a1 = 0.9, bias = 0):
        start_time_training = time.time()
        record_reservoir_train_nrow = int(self.train_true.shape[0] + 1) # size of the training data
        self.record_reservoir_nodes = np.zeros((record_reservoir_train_nrow, self.num_reservoir_nodes))
        I = np.identity(self.record_reservoir_nodes.shape[1])
        self.const_input = const_input
        self.C1 = C1
        self.a1 = a1
        self.leak_rate = (1-C1*a1)
        self.bias = bias
        
        # Calculate the next state
        for data_i, train_back in enumerate(self.train_true):
            # Set the current state
            x0 = self.record_reservoir_nodes[data_i,:]
            u1 = np.array([self.const_input])
            y0 = np.array([train_back])
            x1 = (1 - self.C1 * self.a1) * x0 + self.C1 * (np.tanh(u1 @ self.Win + x0 @ self.W + y0 @ self.Wback + self.bias))
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
        self.test_reservoir_nodes = np.zeros((record_reservoir_test_nrow, self.num_reservoir_nodes))
        self.test_reservoir_nodes[0,:] = self.record_reservoir_nodes[-1]
        I = np.identity(self.record_reservoir_nodes.shape[1])
        
        for data_i, test_back in enumerate(self.test_true):
            # Set the current state
            x0 = self.test_reservoir_nodes[data_i,:]
            u1 = np.array([self.const_input])
            concat_reservoir_nodes = np.hstack([self.test_reservoir_nodes[data_i,:], self.const_input])
            y0 = np.array([concat_reservoir_nodes @ self.Wout])
            x1 = (1 - self.C1 * self.a1) * x0 + self.C1 * (np.tanh(u1 @ self.Win + x0 @ self.W + y0 @ self.Wback + self.bias))
            self.test_reservoir_nodes[data_i + 1,:] = x1
        
        #self.test_reservoir_nodes = self.test_reservoir_nodes[1:,]
        washed_nrow = self.test_reservoir_nodes.shape[0]
        const_input_mat = np.repeat(self.const_input, washed_nrow).reshape(washed_nrow,1)
        self.test_reservoir_nodes = np.hstack([self.test_reservoir_nodes, const_input_mat]) # Remove the last row
        self.test_predicted = self.test_reservoir_nodes[:-1,:] @ self.Wout
        self.test_pred = np.corrcoef([self.test_true], [self.test_predicted])[1,0]
        self.test_rmse = np.sqrt(np.mean((np.array(self.test_true) - self.test_predicted)**2))
        self.test_nmse = sum((np.array(self.test_true) - self.test_predicted)**2)/sum(np.array(self.test_true)**2)
        self.testing_time = time.time() - start_time_testing
    
    def summarize_stat(self):
        # Summary statistics
        result_summary = np.array([self.network, round(self.train_pred, 4), round(self.test_pred, 4),
                                 round(self.train_rmse, 4), round(self.test_rmse, 4),
                                 round(self.train_nmse, 7), round(self.test_nmse, 7),
                                 self.num_reservoir_nodes, self.w_sparsity,
                                 self.w_in_strength, self.w_in_sparsity,
                                 self.alpha, self.w_back_sparsity, self.w_back_strength,
                                 round(self.leak_rate, 2), len(self.train_data) + len(self.test_data),
                                 self.test_fraction, self.washout,
                                 round(self.training_time, 2), round(self.learnmodel_time, 2), round(self.testing_time, 2)])
        self.result_summary_df = pd.DataFrame(result_summary.reshape(1,21))
        self.result_summary_df = self.result_summary_df.rename(columns = {0:"network",
                                                          1:"train_pred", 2:"test_pred",
                                                          3:"RMSE_train", 4:"RMSE_test",
                                                          5:"NMSE_train", 6:"NMSE_test",
                                                          7:"num_nodes",  8:"W_sparsity",
                                                          9:"Win_strength", 10:"Win_sparsity",
                                                          11:"alpha",
                                                          12:"Wback_sparsity", 13:"Wback_strength",
                                                          14:"leak_rate", 15:"data_size",
                                                          16:"test_fraction", 17:"washout_data",
                                                          18:"training_time", 19:"learnmodel_time", 20:"testing_time"})
#=========================================================================#
