####
#### Ecological Reservoir Computing with python
#### 2020.4.17, Ushio
####
#### 2020.4.17: First implementation
#### 2020.4.19: Revised for time series prediction
####

# Load modules
import numpy as np
import pandas as pd
import time
#from scipy import linalg
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
#from sklearn.metrics import accuracy_score

# Define functions
### Embedding function ---------------------------------------------------------------------------#
def embed(ts_data, E = 2, tau = 1):
    ts1 = ts_data
    if E > 1:
        # Time-delay embedding to reconstruct state-space
        for i in range(E-1):
            nan_ts = np.repeat(np.nan, (i+1)*tau)
            if i == 0:
                delay_ts = pd.concat([pd.Series(nan_ts), ts1.iloc[0:-(i+1)*tau,]], axis = 0)
            else:
                delay_ts = pd.concat([pd.Series(nan_ts), ts1.iloc[0:-(i+1)*tau,0]], axis = 0)
            delay_ts.index = pd.Index(range(ts1.shape[0]))
            ts1 = pd.concat([ts1, delay_ts], axis = 1)
        # Remove all rows containing any NaN
        ts1.columns = pd.Index(range(ts1.shape[1]))
        return ts1.values
    else:
        return np.array(ts1)
#---------------------------------------------------------------------------#

# Define class "SimplexReservoir"
class rtERC(): #=========================================================================#
    # Step 1: Initialize class "SimplexReservoir"
    def __init__(self, reservoir_name):
        self.reservoir_name = reservoir_name
    
    # Step 2: Input reservoir state and learn model
    def learn_model(self, reservoir_state, train_true,
                    washout_fraction = 0.05, ridge_lambda = 0.05):
                    #max_iter = 100, solver = 'newton-cg', penalty = "l2", C = 1):
        start_time_learnmodel = time.time()
        self.ridge_lambda = ridge_lambda
        self.reservoir_state = reservoir_state
        self.train_true = train_true
        data_length = self.reservoir_state.shape[0]
        self.reservoir_state = self.reservoir_state.reshape(data_length, -1)
        self.reservoir_E = self.reservoir_state.shape[1]
        self.train_true = self.train_true.reshape(data_length, -1)
        
        # Identify no NA indices
        reservoir_state_nona_id = pd.isna(self.reservoir_state).sum(axis=1) == False
        train_true_nona_id = pd.isna(self.train_true).sum(axis=1) == False
        nona_id = ~np.any(np.array([~reservoir_state_nona_id, ~train_true_nona_id]), axis = 0)
        
        # Wash out
        self.washout = int(washout_fraction * self.reservoir_state[nona_id,:].shape[0])
        self.reservoir_state_washed = self.reservoir_state[nona_id,:][self.washout:,]
        self.train_true_washed = self.train_true[nona_id][self.washout:,]
        self.train_true_washed = self.train_true_washed.reshape(self.train_true_washed.shape[0],)
        
        # Ridge regression (sklearn)
        self.ridge_model = Ridge(alpha = ridge_lambda)
        self.ridge_model.fit(self.reservoir_state_washed, self.train_true_washed)
        self.train_predicted = self.ridge_model.predict(self.reservoir_state_washed)
        self.train_ridge_score = self.ridge_model.score(self.reservoir_state_washed, self.train_true_washed) # R^2
        self.train_cor = np.corrcoef([self.train_true_washed], [self.train_predicted])[1,0]
        self.train_r2 = self.train_cor ** 2
        self.train_rmse = np.sqrt(np.mean((np.array(self.train_true_washed - self.train_predicted))**2))
        self.train_nmse = sum((np.array(self.train_true_washed) - self.train_predicted)**2)/sum(np.array(self.train_true_washed)**2)
        self.learnmodel_time = time.time() - start_time_learnmodel
    
    def predict(self, test_reservoir_state, test_true):
        start_time_testing = time.time()
        self.test_reservoir_state = test_reservoir_state
        self.test_true = test_true
        data_length = self.test_reservoir_state.shape[0]
        self.test_reservoir_state = self.test_reservoir_state.reshape(data_length, -1)
        
        # Step.5: Exploitation
        self.test_predicted = self.ridge_model.predict(self.test_reservoir_state)
        self.test_ridge_score = self.ridge_model.score(self.test_reservoir_state, self.test_true)
        self.test_cor = np.corrcoef([self.test_true], [self.test_predicted])[1,0]
        self.test_r2 = self.test_cor ** 2
        self.test_rmse = np.sqrt(np.mean((np.array(self.test_true) - self.test_predicted)**2))
        self.test_nmse = sum((np.array(self.test_true) - self.test_predicted)**2)/sum(np.array(self.test_true)**2)
        self.testing_time = time.time() - start_time_testing
    
    # Step. 7: Summarize stats
    def summarize_stat(self):
        # Summary statistics
        result_summary = np.array([self.reservoir_name, self.reservoir_E,
                                 round(self.train_ridge_score, 4),
                                 round(self.train_cor, 4),
                                 round(self.train_r2, 4),
                                 round(self.test_ridge_score, 4),
                                 round(self.test_cor, 4),
                                 round(self.test_r2, 4),
                                 self.washout, round(self.learnmodel_time, 2), round(self.testing_time, 2)])
        self.result_summary_df = pd.DataFrame(result_summary.reshape(1,11))
        self.result_summary_df = self.result_summary_df.rename(columns = {0:"reservoir_name", 1:"reservoir_E",
                                                          2:"train_ridge_score",
                                                          3:"train_cor",
                                                          4:"train_r2",
                                                          5:"test_ridge_score",
                                                          6:"test_cor",
                                                          7:"test_r2",
                                                          8:"washout_data", 9:"learnmodel_time", 10:"testing_time"})
#====================================================================================================#


