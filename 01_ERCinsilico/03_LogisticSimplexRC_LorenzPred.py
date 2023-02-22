####
#### Ecological Reservoir Computing
#### No.3 Simplex reservoir using logistic map (time series)
####

# Import essential modules
import numpy as np # np.__version__ # 1.19.0
import pandas as pd # pd.__version__ # 1.0.5
import pyEDM as edm # 1.2.1.0, 2020.7.8
import itertools; import joblib; import time
from scipy import linalg

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import custom module
import module.predict_by_ecological_reservoir_v1 as erc
#import importlib; importlib.reload(erc)

# Create output directory
import os; output_dir = "03_LogisticSimplexRC_LorenzPredOut"; os.mkdir(output_dir)


# -------------------- Prepare data and set parameters -------------------- #
# Load time series
lorenz = pd.read_csv("./data/LorenzTS_tau10.csv")
logis_ts = pd.read_csv("./data/logistic_Xr2_92Yr2_90.csv")

# Select library data set
reservoir_var, reservoir_db_name, reservoir_ts = "x", "logis_ts", logis_ts[5:].reset_index(drop = True)
target_ts, target_var, target_db_name, test_fraction =  lorenz[91:], 'x', 'lorenz', 0.2

# Select target data
lambda0 = 0.05
test_fraction = 0.2

# ---------- Use of the optimal embedding dimension ----------- #
simp_res = edm.EmbedDimension(dataFrame = reservoir_ts,
                lib = "1 1500", pred = "1 1500",
                maxE = 100,
                columns = reservoir_var,
                target = reservoir_var) 
simp_res.to_csv("%s/SimplexResult.csv" % output_dir)


# ---------- Grid search of the best parameter ----------- #
# Define function for parallel computing
# ------------------------------ For parallel analysis ------------------------------ #
def reservoir_computing_parallel(reservoir_var, reservoir_ts, reservoir_db_name,
                                 target_ts, target_var, target_db_name, test_fraction = 0.2,
                                 n_nodes = 100, w_in_strength = 2, w_in_sparsity = 0.1, n_nn = None):
  par_erc = erc.SimplexReservoir(reservoir_var, reservoir_ts, reservoir_db_name)
  par_erc.compile_reservoir_data(n_nodes)
  par_erc.prepare_target_data(target_ts, target_var, target_db_name, test_fraction)
  par_erc.initialize_reservoir(w_in_strength = w_in_strength, w_in_sparsity = w_in_sparsity, n_nn = n_nn)
  par_erc.compute_reservoir_state(initial_method = "zero")
  par_erc.learn_model(ridge_lambda = 0.05)
  par_erc.predict()
  par_erc.summarize_stat()
  return par_erc.result_summary_df
# ------------------------------------------------------------------------------------------ #

# ------------------------------ For nodes and nn check ------------------------------ #
def check_nodes_nn(reservoir_var, reservoir_ts, reservoir_db_name,
                   target_ts, target_var, target_db_name,
                   node_range = [10,20,30], nn_i = None):
  output_nodes_nn = pd.DataFrame()
  for nodes_i in node_range:
    src_res = reservoir_computing_parallel(reservoir_var, reservoir_ts, reservoir_db_name,
                                           target_ts, target_var, target_db_name,
                                           n_nodes = nodes_i, n_nn = nn_i)
    # Append summary dataframe
    output_nodes_nn = output_nodes_nn.append(src_res)
  return output_nodes_nn
# ------------------------------------------------------------------------------------------ #

# Set search parameters
node_range = [np.arange(10,101,10).tolist(), np.arange(200,501,100).tolist()]; node_range = sum(node_range, [])
nn_range = [np.arange(2,32,1).tolist(), [51, 101, None]]; nn_range = sum(nn_range, [])

par_rc1 = joblib.Parallel(n_jobs=-3, verbose = 10)([joblib.delayed(check_nodes_nn)
                         (reservoir_var, reservoir_ts, reservoir_db_name, target_ts, target_var, target_db_name,
                          node_range = node_range, nn_i = x) for x in nn_range])
output_all_df1 = par_rc1[0]
for i in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[i])
output_all_df1 = output_all_df1.reset_index(drop = True)

# Check the best parameters
def CheckPerformance(output_df):
  min_rmse_id = pd.Series(output_df['NMSE_test'], dtype='float').idxmin()
  return output_df.iloc[min_rmse_id]

CheckPerformance(output_all_df1)
best_nodes = int(CheckPerformance(output_all_df1)['num_nodes'])
best_nn = int(CheckPerformance(output_all_df1)['n_neighbors'])


# Check best w_in_sparsity and w_in_strength
WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WinSt_range = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

# Multiprocess computing
par_rc2 = joblib.Parallel(n_jobs=-3, verbose = 10)([joblib.delayed(reservoir_computing_parallel)
                         (reservoir_var, reservoir_ts, reservoir_db_name,
                          target_ts, target_var, target_db_name, test_fraction = 0.2,
                          n_nodes = best_nodes, w_in_sparsity = x, w_in_strength = y, n_nn = best_nn) for x, y in itertools.product(WinSp_range, WinSt_range)])
output_all_df2 = par_rc2[0]
for j in range(1,len(par_rc2)): output_all_df2 = output_all_df2.append(par_rc2[j])
output_all_df2 = output_all_df2.reset_index(drop = True)
w_in_sp_min = output_all_df2.loc[output_all_df2['NMSE_test'].min() == output_all_df2['NMSE_test'],"Win_sparsity"]
w_in_st_min = output_all_df2.loc[output_all_df2['NMSE_test'].min() == output_all_df2['NMSE_test'],"Win_strength"]
w_in_sp_best = float(np.array(w_in_sp_min)[0])
w_in_st_best = float(np.array(w_in_st_min)[0])

# Save best parameters
output_all_df1 = output_all_df1.reset_index(drop = True); output_all_df1.to_csv('%s/LSR_output_df1.csv' % output_dir)
output_all_df2 = output_all_df2.reset_index(drop = True); output_all_df2.to_csv('%s/LSR_output_df2.csv' % output_dir)



# ---------- Perform the best result ---------- #
erc1 = erc.SimplexReservoir(reservoir_var, reservoir_ts, reservoir_db_name)
erc1.compile_reservoir_data(100)
erc1.prepare_target_data(target_ts, target_var, target_db_name, test_fraction)
erc1.initialize_reservoir(w_in_strength = w_in_st_best, w_in_sparsity = w_in_sp_best)
erc1.compute_reservoir_state(initial_method = "zero")
erc1.learn_model(ridge_lambda = lambda0)
erc1.predict()
erc1.summarize_stat()
erc1.result_summary_df

# Save result to pickel
joblib.dump(erc1, "%s/LSR_PredLorenz.jb" % output_dir, compress=3)
