####
#### Reservoir computing
#### No.9 Emulate Mackey-Glass in closed-loop by ESN
####

# Import essential modules
import numpy as np
import pandas as pd
import itertools; import joblib; import time
from scipy import linalg

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import custom module
import module.closedloop_by_random_reservoir_v1 as cloop

# Create output directory
import os; output_dir = "09_ESN_MackeyGlassOut"; os.mkdir(output_dir)


# -------------------- Prepare data and set parameters -------------------- #
# Load time series
mackey_ts = pd.read_csv("./data/mackey_glass17.csv").iloc[1000:7000,]

# Standdardize mackey_ts
train_data_size = 5400
mackey_std = (mackey_ts["value"] - mackey_ts["value"].min())/(mackey_ts["value"].max() - mackey_ts["value"].min())
train_true = mackey_std.iloc[0:train_data_size]
test_true = mackey_std.iloc[train_data_size:]
#train_data = None # use constant input
#test_data = None # use constant input
train_data = mackey_ts["random_input"].iloc[0:train_data_size] # Dummy, constant input is used in the function
test_data = mackey_ts["random_input"].iloc[train_data_size:] # Dummy, constant input is used in the function

# ------------------------------ For parallel analysis ------------------------------ #
def cloop_parallel(n_nodes, alpha, train_data, train_true, test_data, test_true,
                   w_in_sparsity = 0, w_in_strength = 0.1, w_sparsity = 0,
                   w_back_sparsity = 0, w_back_strength = 1.19,
                   Win_seed = 1234, W_seed = 1235, Wback_seed = 1236, return_obj = "summary"):
  # Step.1: Initialize class "SimplexReservoir"
  par_rrc = cloop.RandomReservoir(network_name = "Mackey_Glass_closed_loop")
  # Step.2: Prepare target data
  par_rrc.prepare_data(train_data, train_true, test_data, test_true,
                      train_var = "Mackey_Glass", test_var = "None")
  # Step.3: Initialize reservoir
  par_rrc.initialize_reservoir(n_nodes, alpha = alpha,
                         w_in_sparsity = w_in_sparsity,
                         w_in_strength = w_in_strength, 
                         w_sparsity = w_sparsity,
                         w_back_sparsity = w_back_sparsity,
                         w_back_strength = w_back_strength,
                         Win_seed = Win_seed, W_seed = W_seed, Wback_seed = Wback_seed)
  # Step.4: Training reservoir
  par_rrc.compute_reservoir_state(const_input = 0.2, C1 = 0.44, a1 = 0.9)
  # Step.5: Learn weights by ridge regression
  par_rrc.learn_model(ridge_lambda = 0.05, washout_fraction = 0)
  # Step.6: Predict test data
  par_rrc.predict()
  # Step.7: Summarize stats
  par_rrc.summarize_stat()
  par_rrc.result_summary_df
  if return_obj == "summary":
    return par_rrc.result_summary_df
  else:
    return par_rrc
# ------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# Parameter search
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# The range of number of nodes
node_range = [[1,2,3,4,5,6,7,8,9,10,15], np.arange(20,201,20).tolist(), np.arange(200,1001,50).tolist()]
node_range = sum(node_range, [])
alpha_range = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1, 1.01, 1.05, 1.1, 1.3, 1.5]
WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WinSt_range = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
# Back parameters
WbSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WbSt_range = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

# Multiprocess computing
par_rc1 = joblib.Parallel(n_jobs=-1)([joblib.delayed(cloop_parallel)
                                     (x, 0.9, train_data, train_true, test_data, test_true) for x in node_range])
par_rc2 = joblib.Parallel(n_jobs=-1)([joblib.delayed(cloop_parallel)
                                     (500, x, train_data, train_true, test_data, test_true) for x in alpha_range])
par_rc3 = joblib.Parallel(n_jobs=-1, verbose = 10)([joblib.delayed(cloop_parallel)
                         (500, 0.9, train_data, train_true, test_data, test_true,
                          w_in_sparsity = x, w_in_strength = y, w_sparsity = z) for x, y, z in itertools.product(WinSp_range, WinSt_range, WSp_range)])
par_rc4 = joblib.Parallel(n_jobs=-1, verbose = 10)([joblib.delayed(cloop_parallel)
                         (500, 0.9, train_data, train_true, test_data, test_true,
                          w_back_sparsity = x, w_back_strength = y) for x, y in itertools.product(WbSp_range, WbSt_range)])
# Compile output
output_all_df1 = par_rc1[0]
for i in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[i])
output_all_df2 = par_rc2[0]
for i in range(1,len(par_rc2)): output_all_df2 = output_all_df2.append(par_rc2[i])
output_all_df3 = par_rc3[0]
for i in range(1,len(par_rc3)): output_all_df3 = output_all_df3.append(par_rc3[i])
output_all_df4 = par_rc4[0]
for i in range(1,len(par_rc4)): output_all_df4 = output_all_df4.append(par_rc4[i])
# Reset indices
output_all_df1 = output_all_df1.reset_index(drop = True); output_all_df1.to_csv('%s/ESN_output_df1.csv' % output_dir)
output_all_df2 = output_all_df2.reset_index(drop = True); output_all_df2.to_csv('%s/ESN_output_df2.csv' % output_dir)
output_all_df3 = output_all_df3.reset_index(drop = True); output_all_df3.to_csv('%s/ESN_output_df3.csv' % output_dir)
output_all_df4 = output_all_df4.reset_index(drop = True); output_all_df4.to_csv('%s/ESN_output_df3.csv' % output_dir)
## Extract parameters
num_nodes_min = output_all_df1.loc[output_all_df1['NMSE_test'].min() == output_all_df1['NMSE_test'],"num_nodes"]
alpha_min = output_all_df2.loc[output_all_df2['NMSE_test'].min() == output_all_df2['NMSE_test'],"alpha"]
w_in_sp_min = output_all_df3.loc[output_all_df3['NMSE_test'].min() == output_all_df3['NMSE_test'],"Win_sparsity"]
w_in_st_min = output_all_df3.loc[output_all_df3['NMSE_test'].min() == output_all_df3['NMSE_test'],"Win_strength"]
w_sp_min = output_all_df3.loc[output_all_df3['NMSE_test'].min() == output_all_df3['NMSE_test'],"W_sparsity"]
w_bc_sp_min = output_all_df4.loc[output_all_df4['NMSE_test'].min() == output_all_df4['NMSE_test'],"Wback_sparsity"]
w_bc_st_min = output_all_df4.loc[output_all_df4['NMSE_test'].min() == output_all_df4['NMSE_test'],"Wback_strength"]
# Specify the best value
n_nodes_best = 500
alpha_best = float(np.array(alpha_min)[0])
w_in_sp_best = float(np.array(w_in_sp_min)[0])
w_in_st_best = float(np.array(w_in_st_min)[0])
w_sp_best = float(np.array(w_sp_min)[0])
w_bc_sp_best = float(np.array(w_bc_sp_min)[0])
w_bc_st_best = float(np.array(w_bc_st_min)[0])
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #


# Save the best results
import importlib; importlib.reload(cloop)
cloop_01 = cloop.RandomReservoir(network_name = "Mackey_Glass_closed_loop")
cloop_01.prepare_data(train_data, train_true, test_data, test_true,
                      train_var = "Mackey_Glass", test_var = "None")
cloop_01.initialize_reservoir(n_nodes_best, alpha = alpha_best,
                         w_in_sparsity = w_in_sp_best,
                         w_in_strength = w_in_st_best,
                         w_sparsity = w_sp_best,
                         w_back_sparsity = w_bc_sp_best,
                         w_back_strength = w_bc_st_best,
                         Win_seed = 1234, W_seed = 1235, Wback_seed = 1236)
cloop_01.compute_reservoir_state(const_input = 0.2, C1 = 0.44, a1 = 0.9)
cloop_01.learn_model(ridge_lambda = 0.05, washout_fraction = 0.05)
cloop_01.predict()
cloop_01.summarize_stat()
cloop_01.result_summary_df
cloop_01.result_summary_df.to_csv('%s/ESN_MackeyGlass_Best.csv' % output_dir)

# Save result to pickel
joblib.dump(cloop_01, "%s/ESN_MackeyGlass.jb" % (output_dir), compress=3)


exit
library(reticulate)

plot(py$cloop_01$test_true, type = "l", ylim = c(0,1), xlim = c(0,800))
lines(py$cloop_01$test_predicted, col = 2)
