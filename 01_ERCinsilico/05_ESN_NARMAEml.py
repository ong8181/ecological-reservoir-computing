####
#### Ecological Reservoir Computing
#### No.5 Emulate NARMA by ESN
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
import module.emulate_by_random_reservoir_v1 as err

# Create output directory
import os; output_dir = "05_ESN_NARMAEmlOut"; os.mkdir(output_dir)

# -------------------- Prepare data and set parameters -------------------- #
# Load time series
narma02 = pd.read_csv('./data/NARMA02.csv').iloc[6000:7000,]
narma03 = pd.read_csv('./data/NARMA03.csv').iloc[6000:7000,]
narma04 = pd.read_csv('./data/NARMA04.csv').iloc[6000:7000,]
narma05 = pd.read_csv('./data/NARMA05.csv').iloc[6000:7000,]
narma10 = pd.read_csv('./data/NARMA10.csv').iloc[6000:7000,]

## Change target_db_name to analyze different NARMA
# Set target variable
target_db_name = "narma05"
target_ts = narma05.iloc[:-1,:]
true_ts = narma05.iloc[1:,:]

target_var = 'input'
true_var = 'value'
test_fraction = 0.5

# ------------------------------ For parallel analysis ------------------------------ #
def reservoir_computing_parallel(n_nodes, alpha, w_in_sparsity = 0.1, w_in_strength = 1,
                                 w_sparsity = 0.1, leak_rate = 0,
                                 Win_seed = 1234, W_seed = 1235, return_obj = "summary"):
  par_err = err.RandomReservoir(network_name = "ESN: %s" % target_db_name)
  par_err.prepare_target_data(target_ts, target_var, true_ts, true_var, test_fraction = test_fraction)
  par_err.initialize_reservoir(n_nodes, alpha,
                           w_in_sparsity = w_in_sparsity, w_in_strength = w_in_strength,
                           w_sparsity = w_sparsity, leak_rate = leak_rate,
                           Win_seed = Win_seed, W_seed = W_seed)
  par_err.compute_reservoir_state()
  par_err.learn_model(ridge_lambda = 0.05, washout_fraction = 0.05)
  par_err.predict()
  par_err.summarize_stat()
  if return_obj == "summary":
    return par_err.result_summary_df
  else:
    return par_err
# ------------------------------------------------------------------------------------------ #
# Initial search
# The range of number of nodes
node_range = [[1], np.arange(20,1001,20).tolist(), np.arange(1000,2001,100).tolist()]
node_range = sum(node_range, [])
alpha_range = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1, 1.01, 1.05, 1.1, 1.3, 1.5]
Leak_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
## For grid search
WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WinSt_range = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
WSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Multiprocess computing
par_rc1 = joblib.Parallel(n_jobs=-1)([joblib.delayed(reservoir_computing_parallel)(x, alpha = 0.95) for x in node_range])
par_rc2 = joblib.Parallel(n_jobs=-1)([joblib.delayed(reservoir_computing_parallel)(1000, alpha = x) for x in alpha_range])
par_rc3 = joblib.Parallel(n_jobs=-1)([joblib.delayed(reservoir_computing_parallel)(1000, alpha = 0.95, leak_rate = x) for x in Leak_range])
par_rc4 = joblib.Parallel(n_jobs=-1, verbose = 10)([joblib.delayed(reservoir_computing_parallel)
                         (1000, alpha = 0.95, w_in_sparsity = x, w_in_strength = y, w_sparsity = z) for x, y, z in itertools.product(WinSp_range, WinSt_range, WSp_range)])
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
output_all_df1 = output_all_df1.reset_index(drop = True); output_all_df1.to_csv('%s/ESN_%s_df1.csv' % (output_dir, target_db_name))
output_all_df2 = output_all_df2.reset_index(drop = True); output_all_df2.to_csv('%s/ESN_%s_df2.csv' % (output_dir, target_db_name))
output_all_df3 = output_all_df3.reset_index(drop = True); output_all_df3.to_csv('%s/ESN_%s_df3.csv' % (output_dir, target_db_name))
output_all_df4 = output_all_df4.reset_index(drop = True); output_all_df4.to_csv('%s/ESN_%s_df4.csv' % (output_dir, target_db_name))


# ---------------------------------------------- #
## Function to extract best parameters
def extract_parms(parm_df_name):
  parm_df1 = pd.read_csv('05_ESN_NARMAEmlOut/%s_df1.csv' % parm_df_name)
  parm_df2 = pd.read_csv('05_ESN_NARMAEmlOut/%s_df2.csv' % parm_df_name)
  parm_df3 = pd.read_csv('05_ESN_NARMAEmlOut/%s_df3.csv' % parm_df_name)
  parm_df4 = pd.read_csv('05_ESN_NARMAEmlOut/%s_df4.csv' % parm_df_name)
  best_n_node = 2000 # fixed
  best_alpha = float(parm_df2.iloc[parm_df2["NMSE_test"].astype("float64").idxmin(),].loc["alpha"])
  best_leak_rate = float(parm_df3.iloc[parm_df3["NMSE_test"].astype("float64").idxmin(),].loc["leak_rate"])
  best_w_in_sparsity = float(parm_df4.iloc[parm_df4["NMSE_test"].astype("float64").idxmin(),].loc["Win_sparsity"])
  best_w_in_strength = float(parm_df4.iloc[parm_df4["NMSE_test"].astype("float64").idxmin(),].loc["Win_strength"])
  best_w_sparsity = float(parm_df4.iloc[parm_df4["NMSE_test"].astype("float64").idxmin(),].loc["W_sparsity"])
  # Return parameters
  return [best_n_node, best_alpha, best_leak_rate, best_w_in_sparsity, best_w_in_strength, best_w_sparsity]
# ---------------------------------------------- #


# ---------------------------------------------- #
# NARMA02
narma_name = "narma02"
if narma_name == "narma02":
  target_db_name = "narma02"
  target_var, true_var, test_fraction = 'input', 'value', 0.5
  par_ext = extract_parms("ESN_%s" % narma_name)
  rrc2 = err.RandomReservoir(network_name = "Random Network")
  rrc2.prepare_target_data(narma02.iloc[:-1,:], target_var, narma02.iloc[1:,:], true_var, test_fraction = 0.5)
  rrc2.initialize_reservoir(num_reservoir_nodes = par_ext[0], alpha = par_ext[1],
                           w_in_sparsity = par_ext[3], w_in_strength = par_ext[4],
                           w_sparsity = par_ext[5], leak_rate = par_ext[2])
  rrc2.compute_reservoir_state()
  rrc2.learn_model(ridge_lambda = 0.05, washout_fraction = 0)
  rrc2.predict()
  rrc2.summarize_stat()
  rrc2.result_summary_df
  # Save result to pickel
  joblib.dump(rrc2, "%s/ESN_Eml_%s.jb" % (output_dir, target_db_name), compress=3)

# NARMA03
narma_name = "narma03"
if narma_name == "narma03":
  target_db_name = narma_name
  target_var, true_var, test_fraction = 'input', 'value', 0.5
  par_ext = extract_parms("ESN_%s" % narma_name)
  rrc3 = err.RandomReservoir(network_name = "Random Network")
  rrc3.prepare_target_data(narma03.iloc[:-1,:], target_var, narma03.iloc[1:,:], true_var, test_fraction = 0.5)
  rrc3.initialize_reservoir(num_reservoir_nodes = par_ext[0], alpha = par_ext[1],
                           w_in_sparsity = par_ext[3], w_in_strength = par_ext[4],
                           w_sparsity = par_ext[5], leak_rate = par_ext[2])
  rrc3.compute_reservoir_state()
  rrc3.learn_model(ridge_lambda = 0.05, washout_fraction = 0)
  rrc3.predict()
  rrc3.summarize_stat()
  rrc3.result_summary_df
  # Save result to pickel
  joblib.dump(rrc3, "%s/ESN_Eml_%s.jb" % (output_dir, target_db_name), compress=3)

# NARMA04
narma_name = "narma04"
if narma_name == "narma04":
  target_db_name = narma_name
  target_var, true_var, test_fraction = 'input', 'value', 0.5
  par_ext = extract_parms("ESN_%s" % narma_name)
  rrc4 = err.RandomReservoir(network_name = "Random Network")
  rrc4.prepare_target_data(narma04.iloc[:-1,:], target_var, narma04.iloc[1:,:], true_var, test_fraction = 0.5)
  rrc4.initialize_reservoir(num_reservoir_nodes = par_ext[0], alpha = par_ext[1],
                           w_in_sparsity = par_ext[3], w_in_strength = par_ext[4],
                           w_sparsity = par_ext[5], leak_rate = par_ext[2])
  rrc4.compute_reservoir_state()
  rrc4.learn_model(ridge_lambda = 0.05, washout_fraction = 0)
  rrc4.predict()
  rrc4.summarize_stat()
  rrc4.result_summary_df
  # Save result to pickel
  joblib.dump(rrc4, "%s/ESN_Eml_%s.jb" % (output_dir, target_db_name), compress=3)

# NARMA05
narma_name = "narma05"
if narma_name == "narma05":
  target_db_name = narma_name
  target_var, true_var, test_fraction = 'input', 'value', 0.5
  par_ext = extract_parms("ESN_%s" % narma_name)
  rrc5 = err.RandomReservoir(network_name = "Random Network")
  rrc5.prepare_target_data(narma05.iloc[:-1,:], target_var, narma05.iloc[1:,:], true_var, test_fraction = 0.5)
  rrc5.initialize_reservoir(num_reservoir_nodes = par_ext[0], alpha = par_ext[1],
                           w_in_sparsity = par_ext[3], w_in_strength = par_ext[4],
                           w_sparsity = par_ext[5], leak_rate = par_ext[2])
  rrc5.compute_reservoir_state()
  rrc5.learn_model(ridge_lambda = 0.05, washout_fraction = 0)
  rrc5.predict()
  rrc5.summarize_stat()
  rrc5.result_summary_df
  # Save result to pickel
  joblib.dump(rrc5, "%s/ESN_Eml_%s.jb" % (output_dir, target_db_name), compress=3)

# NARMA10
narma_name = "narma10"
if narma_name == "narma10":
  target_db_name = narma_name
  target_var, true_var, test_fraction = 'input', 'value', 0.5
  par_ext = extract_parms("ESN_%s" % narma_name)
  rrc = err.RandomReservoir(network_name = "Random Network")
  rrc.prepare_target_data(narma10.iloc[:-1,:], target_var, narma10.iloc[1:,:], true_var, test_fraction = 0.5)
  rrc.initialize_reservoir(num_reservoir_nodes = par_ext[0], alpha = par_ext[1],
                           w_in_sparsity = par_ext[3], w_in_strength = par_ext[4],
                           w_sparsity = par_ext[5], leak_rate = par_ext[2])
  rrc.compute_reservoir_state()
  rrc.learn_model(ridge_lambda = 0.05, washout_fraction = 0)
  rrc.predict()
  rrc.summarize_stat()
  rrc.result_summary_df
  # Save result to pickel
  joblib.dump(rrc, "%s/ESN_Eml_%s.jb" % (output_dir, target_db_name), compress=3)
