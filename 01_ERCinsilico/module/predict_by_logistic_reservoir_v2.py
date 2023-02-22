####
#### Logistic reservoir, prediction
####

# Load modules
import numpy as np
import pandas as pd
import time
from scipy import linalg

class LogisticReservoir(): #=========================================================================#
    # Step 1: Initialize class "RandomReservoir"
    def __init__(self, network_name = "Random_Network"):
        self.network = network_name
    
    # Step 2: Prepare target data
    def prepare_target_data(self, target_ts, target_var, test_fraction):
        self.target_data  = target_ts[target_var]
        self.target_var = target_var
        self.test_fraction = test_fraction
        # Standardize data
        std_data = (self.target_data - self.target_data.min()) / (self.target_data.max() - self.target_data.min())
        
        # Split data into training data set and test data set
        data_size = self.target_data.shape[0]
        test_size = int(round(data_size*self.test_fraction))
        train_size = int(data_size - test_size)
        self.train_data = std_data[0:train_size].reset_index()[self.target_var]
        self.train_true = std_data[1:(train_size + 1)].reset_index()[self.target_var]
        self.test_data = std_data[train_size:(data_size - 1)].reset_index()[self.target_var]
        self.test_true = std_data[(train_size + 1):data_size].reset_index()[self.target_var]
    
    # Step 3: Initialize reservoir
    def initialize_reservoir(self, num_reservoir_nodes = 2, w_in_sparsity = 0, w_in_strength = 0.05, num_input_nodes = 1, num_output_nodes = 1):
        # Set parameters
        self.num_reservoir_nodes = num_reservoir_nodes
        self.w_in_sparsity = w_in_sparsity
        self.w_in_strength = w_in_strength
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        
        # Initialize W_in
        np.random.seed(1234); W_in0 = np.random.uniform(-1, 1, (self.num_input_nodes * self.num_reservoir_nodes, 1))
        np.random.seed(1234); rand_id = np.random.choice(W_in0.shape[0], int(self.num_input_nodes * self.num_reservoir_nodes * self.w_in_sparsity), replace = False)
        W_in0[rand_id] = 0; self.W_in = W_in0.reshape(self.num_input_nodes, self.num_reservoir_nodes) * self.w_in_strength
    
    # Step 4: Compute reservoir states
    def compute_reservoir_state(self, rx = 3.8, ry = 3.5, bxy = 0.1, byx = 0.02, initial_state = [0.5, 0.5]):
        start_time_training = time.time()
        self.rx, self.ry, self.bxy, self.byx = rx, ry, bxy, byx
        record_reservoir_train_nrow = int(self.train_data.shape[0] + 1) # size of the training data
        self.record_reservoir_nodes = np.zeros((record_reservoir_train_nrow, self.num_reservoir_nodes))
        
        # Set initial state
        self.record_reservoir_nodes[0,:] = initial_state
        
        # Calculate the next state
        for data_i, input_train in enumerate(self.train_data):
            x0, y0 = self.record_reservoir_nodes[data_i,:]
            # Logistic equation
            x1 = x0*(self.rx - self.rx * x0 - self.byx * y0)
            y1 = y0*(self.ry - self.bxy * x0 - self.ry * y0)
            # Add input vector
            x_n1 = [input_train] @ self.W_in + [x1, y1]
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
    
    def learn_model_wo_reservoir(self, washout_fraction = 0, ridge_lambda = 0.05):
        start_time_learnmodel = time.time()
        self.ridge_lambda = ridge_lambda
        self.washout = int(washout_fraction * self.train_data.shape[0])
        train_data_washed = self.train_data[self.washout:,]
        # Ridge Regression
        E_lambda = self.ridge_lambda
        inv_x = 1/(train_data_washed.T @ train_data_washed + E_lambda)
        # update weights of output layer
        self.W_out = (inv_x * train_data_washed.T) @ np.array(self.train_true[self.washout:])
        self.train_predicted = train_data_washed * self.W_out
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
            x0, y0 = self.test_reservoir_nodes[data_i,:]
            # Logistic equation
            x1 = x0*(self.rx - self.rx * x0 - self.byx * y0)
            y1 = y0*(self.ry - self.bxy * x0 - self.ry * y0)
            # Add input vector
            x_n1 = [input_test] @ self.W_in + [x1, y1]
            self.test_reservoir_nodes[data_i + 1,:] = x_n1
        
        self.test_reservoir_nodes = self.test_reservoir_nodes[1:,]
        self.test_predicted = self.test_reservoir_nodes @ self.W_out
        self.test_pred = np.corrcoef([self.test_true], [self.test_predicted])[1,0]
        self.test_rmse = np.sqrt(np.mean((self.test_predicted - np.array(self.test_true))**2))
        self.test_nmse = sum((np.array(self.test_true) - self.test_predicted)**2)/sum(np.array(self.test_true)**2)
        self.testing_time = time.time() - start_time_testing
    
    def predict_wo_reservoir(self):
        start_time_testing = time.time()
        self.test_predicted = self.test_data * self.W_out
        self.test_pred = np.corrcoef([self.test_true], [self.test_predicted])[1,0]
        self.test_rmse = np.sqrt(np.mean((self.test_predicted - np.array(self.test_true))**2))
        self.test_nmse = sum((np.array(self.test_true) - self.test_predicted)**2)/sum(np.array(self.test_true)**2)
        self.testing_time = time.time() - start_time_testing
    
    def summarize_stat(self):
        # Summary statistics
        result_summary = np.array([self.network, round(self.train_pred, 4), round(self.test_pred, 4),
                                 round(self.train_rmse, 4), round(self.test_rmse, 4),
                                 round(self.train_nmse, 7), round(self.test_nmse, 7),
                                 round(self.rx, 3), round(self.ry, 3), round(self.bxy, 3), round(self.byx, 3),
                                 self.num_reservoir_nodes, self.w_in_strength, self.w_in_sparsity,
                                 self.target_data.shape[0],
                                 self.test_fraction, self.washout,
                                 round(self.training_time, 2), round(self.learnmodel_time, 2), round(self.testing_time, 2)])
        self.result_summary_df = pd.DataFrame(result_summary.reshape(1,20))
        self.result_summary_df = self.result_summary_df.rename(columns = {0:"network",
                                                          1:"rho_train", 2:"rho_test",
                                                          3:"RMSE_train", 4:"RMSE_test",
                                                          5:"NMSE_train", 6:"NMSE_test",
                                                          7:"rx", 8:"ry", 9:"bxy", 10:"byx",
                                                          11:"num_nodes", 12:"Win_strength",
                                                          13:"Win_sparsity", 14:"data_size",
                                                          15:"test_fraction", 16:"washout_data",
                                                          17:"training_time", 18:"learnmodel_time", 19:"testing_time"})
#=========================================================================#
