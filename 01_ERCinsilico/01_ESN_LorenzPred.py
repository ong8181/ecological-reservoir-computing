####
#### Reservoir computing
#### No.1 Echo state network for time series prediction
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
import module.predict_by_random_reservoir_v1 as rrt

# Create output directory
import os; output_dir = "01_ESN_LorenzPredOut"; os.mkdir(output_dir)


# -------------------- Prepare data and set parameters -------------------- #
# Load time series
lorenz = pd.read_csv('./data/LorenzTS_tau10.csv')

# Set target variable
target_var, target_ts = 'x', lorenz[91:]
test_fraction = 0.2

# ------------------------------ For parallel analysis ------------------------------ #
def reservoir_computing_parallel(n_nodes, alpha, w_in_sparsity = 0, w_in_strength = 2, w_sparsity = 0.5,
                                 Win_seed = 1234, W_seed = 1235, return_obj = "summary"):
  # Step.1: Initialize class "SimplexReservoir"
  par_rrc = rrt.RandomReservoir(network_name = "Random Network")
  # Step.2: Prepare target data
  par_rrc.prepare_target_data(target_ts, target_var, test_fraction)
  # Step.3: Initialize reservoir
  par_rrc.initialize_reservoir(n_nodes, alpha,
                           w_in_sparsity = w_in_sparsity, w_in_strength = w_in_strength,
                           w_sparsity = w_sparsity, Win_seed = Win_seed, W_seed = W_seed)
  # Step.4: Training reservoir
  par_rrc.compute_reservoir_state()
  # Step.5: Learn weights by ridge regression
  par_rrc.learn_model(ridge_lambda = 0.05, washout_fraction = 0.05)
  # Step.6: Predict test data
  par_rrc.predict()
  # Step.7: Summarize stats
  par_rrc.summarize_stat()
  if return_obj == "summary":
    return par_rrc.result_summary_df
  else:
    return par_rrc
# ------------------------------------------------------------------------------------------ #

# The range of number of nodes
node_range = [[1,2,3,4,5,6,7,8,9,10,15], np.arange(20,201,20).tolist(), np.arange(200,1001,50).tolist()]
node_range = sum(node_range, [])
alpha_range = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1, 1.01, 1.05, 1.1, 1.3, 1.5]
WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WinSt_range = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

# Multiprocess computing
par_rc1 = joblib.Parallel(n_jobs=-3)([joblib.delayed(reservoir_computing_parallel)(x, alpha = 0.99) for x in node_range])
par_rc2 = joblib.Parallel(n_jobs=-3)([joblib.delayed(reservoir_computing_parallel)(1000, alpha = x) for x in alpha_range])
par_rc3 = joblib.Parallel(n_jobs=-3, verbose = 10)([joblib.delayed(reservoir_computing_parallel)
                         (1000, alpha = 0.99, w_in_sparsity = x, w_in_strength = y, w_sparsity = z) for x, y, z in itertools.product(WinSp_range, WinSt_range, WSp_range)])
# Compile output
output_all_df1 = par_rc1[0]
for i in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[i])
output_all_df2 = par_rc2[0]
for i in range(1,len(par_rc2)): output_all_df2 = output_all_df2.append(par_rc2[i])
output_all_df3 = par_rc3[0]
for i in range(1,len(par_rc3)): output_all_df3 = output_all_df3.append(par_rc3[i])
# Reset indices
output_all_df1 = output_all_df1.reset_index(drop = True); output_all_df1.to_csv('%s/ESN_output_df1.csv' % output_dir)
output_all_df2 = output_all_df2.reset_index(drop = True); output_all_df2.to_csv('%s/ESN_output_df2.csv' % output_dir)
output_all_df3 = output_all_df3.reset_index(drop = True); output_all_df3.to_csv('%s/ESN_output_df3.csv' % output_dir)
## Extract parameters
### NMSE is too small; here we use RMSE
num_nodes_min = output_all_df1.loc[output_all_df1['NMSE_test'].min() == output_all_df1['NMSE_test'],"num_nodes"]
alpha_min = output_all_df2.loc[output_all_df2['NMSE_test'].min() == output_all_df2['NMSE_test'],"alpha"]
w_in_sp_min = output_all_df3.loc[output_all_df3['NMSE_test'].min() == output_all_df3['NMSE_test'],"Win_sparsity"]
w_in_st_min = output_all_df3.loc[output_all_df3['NMSE_test'].min() == output_all_df3['NMSE_test'],"Win_strength"]
w_sp_min = output_all_df3.loc[output_all_df3['NMSE_test'].min() == output_all_df3['NMSE_test'],"W_sparsity"]
#n_nodes_best = int(np.array(num_nodes_min)[0])
alpha_best = float(np.array(alpha_min)[0])
w_in_sp_best = float(np.array(w_in_sp_min)[0])
w_in_st_best = float(np.array(w_in_st_min)[0])
w_sp_best = float(np.array(w_sp_min)[0])


# Best parameters v.s. n_nodes
par_rc0 = joblib.Parallel(n_jobs=1)([joblib.delayed(reservoir_computing_parallel)
                                     (x, alpha = 0.99, w_in_sparsity = w_in_sp_best,
                                     w_in_strength = w_in_st_best, w_sparsity = w_sp_best) for x in node_range])
output_all_df0 = par_rc0[0]
for i in range(1,len(par_rc0)): output_all_df0 = output_all_df0.append(par_rc0[i])
output_all_df0 = output_all_df0.reset_index(drop = True)
output_all_df0.to_csv('%s/SingleNetwork_E1000.csv' % output_dir)


# Save best results
rrc = rrt.RandomReservoir(network_name = "Random Network")
rrc.prepare_target_data(target_ts, target_var, test_fraction)
rrc.initialize_reservoir(num_reservoir_nodes = 1000, alpha = 0.99,
                         w_in_sparsity = w_in_sp_best, w_in_strength = w_in_st_best,
                         w_sparsity = w_sp_best)
rrc.compute_reservoir_state()
rrc.learn_model(ridge_lambda = 0.05, washout_fraction = 0.05)
rrc.predict()
rrc.summarize_stat()
rrc.result_summary_df

# Save result to pickel
joblib.dump(rrc, "%s/ESN_PredLorenz.jb" % output_dir, compress=3)



# ------------------------------------------------------------------------------------------ #
# ------------------------------------- MultiNetowrk approach ------------------------------ #
# ------------------------------------------------------------------------------------------ #
import module.predict_by_multiple_reservoir as msr


for w_node in [3,5,10,20,30,40,50,100]:
  if w_node < 11:
    WRanSeed_range = np.arange(100)
  elif w_node < 51 and w_node >= 11:
    WRanSeed_range = np.arange(100)
  elif w_node < 201 and w_node >= 51:
    WRanSeed_range = np.arange(100)
  
  # Multiprocess computing
  par_rc_random = joblib.Parallel(n_jobs=1)([joblib.delayed(reservoir_computing_parallel)
                                            (w_node, alpha = 0.99, w_in_strength = w_in_st_best, w_in_sparsity = w_in_sp_best,
                                            W_seed = x, return_obj = "class") for x in WRanSeed_range])
  
  output_random = par_rc_random[0].result_summary_df
  for i in range(1,len(par_rc_random)): output_random = output_random.append(par_rc_random[i].result_summary_df)
  # Check maximum prediction skill when single species reservoir is used --------------------- #
  test_pred_all = output_random['rho_test']; test_pred_all = test_pred_all.reset_index(drop = True)
  test_pred_all = test_pred_all.astype("float64"); np.max(test_pred_all); test_pred_all.idxmax()
  
  # Prepare output dataframe
  multinetwork_df = pd.DataFrame()
  
  for n_network in range(0,len(output_random)):
    # Combine reservoir states ----------------------------------------------------------------- #
    combined_reservoir_state = []; combined_test_reservoir_state = [] # Initialize
    if n_network == 0:
          combined_reservoir_state = par_rc_random[0].record_reservoir_nodes
          combined_test_reservoir_state = par_rc_random[0].test_reservoir_nodes
    else:
      for sp_i in range(n_network + 1):
        if len(combined_reservoir_state) == 0:
          combined_reservoir_state = par_rc_random[sp_i].record_reservoir_nodes
          combined_test_reservoir_state = par_rc_random[sp_i].test_reservoir_nodes
        else:
          combined_reservoir_state = np.hstack([combined_reservoir_state, par_rc_random[sp_i].record_reservoir_nodes])
          combined_test_reservoir_state = np.hstack([combined_test_reservoir_state, par_rc_random[sp_i].test_reservoir_nodes])
    # ------------------------------------------------------------------------------------------ #
    
    # Perform multiNetwork analysis
    msr01 = msr.MultinetworkSimplexReservoir("Combined_Network")
    msr01.learn_model(combined_reservoir_state, par_rc_random[0].train_true, ridge_lambda = 1)
    msr01.predict(combined_test_reservoir_state, par_rc_random[0].test_true)
    msr01.summarize_stat()
    msr01.result_summary_df["n_network"] = n_network + 1
    msr01.result_summary_df["E"] = w_node
    msr01.result_summary_df = pd.concat([msr01.result_summary_df, par_rc_random[0].result_summary_df.iloc[:,6:14]], axis = 1)
    multinetwork_df = multinetwork_df.append(msr01.result_summary_df)
  
  multinetwork_df = multinetwork_df.reset_index(drop = True)
  multinetwork_df.to_csv("%s/MultiNetowrk_E%03d.csv" % (output_dir, w_node))

