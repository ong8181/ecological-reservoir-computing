####
#### Multi-ERC prediction
####

# Load modules
import numpy as np
import pandas as pd
import time
from scipy import linalg

# Define class "SimplexReservoir"
class MultinetworkSimplexReservoir(): #=========================================================================#
    # Step 1: Initialize class "SimplexReservoir"
    def __init__(self, network_name):
        self.network = network_name
    
    # Step.2: Learn weights by softmax regression
    def learn_model(self, combined_reservoir_state, train_true, washout_fraction = 0, ridge_lambda = 0.05):
        start_time_learnmodel = time.time()
        self.combined_reservoir_state = combined_reservoir_state
        self.ridge_lambda = ridge_lambda
        self.train_true = train_true
        
        self.washout = int(washout_fraction * self.combined_reservoir_state.shape[0])
        combined_state_washed = self.combined_reservoir_state[self.washout:,]
        
        # Ridge Regression
        E_lambda = np.identity(combined_state_washed.shape[1]) * self.ridge_lambda
        inv_x = np.linalg.inv(combined_state_washed.T @ combined_state_washed + E_lambda)
        # update weights of output layer
        self.W_out = (inv_x @ combined_state_washed.T) @ np.array(self.train_true[self.washout:])
        self.train_predicted = combined_state_washed @ self.W_out
        self.train_pred = np.corrcoef([self.train_true[self.washout:]], [self.train_predicted])[1,0]
        self.train_rmse = np.sqrt(np.mean((self.train_predicted - np.array(self.train_true[self.washout:]))**2))
        self.train_nmse = sum((np.array(self.train_true[self.washout:]) - self.train_predicted)**2)/sum(np.array(self.train_true[self.washout:])**2)
        self.learnmodel_time = time.time() - start_time_learnmodel
    
    # Step.3: Predict test data
    def predict(self, test_reservoir_state, test_true):
        self.test_true = test_true
        self.combined_test_reservoir_state = test_reservoir_state
        # Calculate prediction accuracy for test data
        self.test_predicted = self.combined_test_reservoir_state @ self.W_out
        self.test_pred = np.corrcoef([self.test_true], [self.test_predicted])[1,0]
        self.test_rmse = np.sqrt(np.mean((self.test_predicted - np.array(self.test_true))**2))
        self.test_nmse = sum((np.array(self.test_true) - self.test_predicted)**2)/sum(np.array(self.test_true)**2)
    
    # Step. 4: Summarize stats
    def summarize_stat(self):
        # Summary statistics
        result_summary = np.array([self.network, round(self.train_pred, 4), round(self.test_pred, 4),
                                 round(self.train_rmse, 4), round(self.test_rmse, 4),
                                 round(self.train_nmse, 7), round(self.test_nmse, 7),
                                 self.combined_reservoir_state.shape[1], round(self.learnmodel_time, 2)])
        self.result_summary_df = pd.DataFrame(result_summary.reshape(1,9))
        self.result_summary_df = self.result_summary_df.rename(columns = {0:"network_name",
                                                                          1:"train_pred", 2:"test_pred",
                                                                          3:"RMSE_train", 4:"RMSE_test",
                                                                          5:"NMSE_train", 6:"NMSE_test",
                                                                          7:"total_nodes", 8:"learning_time"})
#====================================================================================================#


