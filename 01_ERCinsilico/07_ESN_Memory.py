####
#### Reservoir computing
#### No.7 Measuring memory capacity of ESN
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
import module.emulate_by_multiple_reservoir_v1 as emr

# Create output directory
import os; output_dir = "07_ESN_MemoryOut"; os.mkdir(output_dir)


# -------------------- Prepare data and set parameters -------------------- #
# Load time series
runif_ts = pd.read_csv("./data/runif_ts.csv").iloc[6000:7000,]


# ------------------------------ For parallel analysis ------------------------------ #
def reservoir_computing_parallel(n_nodes, alpha,
                                 target_ts, target_var, true_ts, true_var,
                                 w_in_sparsity = 0, w_in_strength = 1,
                                 w_sparsity = 0, leak_rate = 0, test_fraction = 0.5,
                                 Win_seed = 1234, W_seed = 1235, return_obj = "summary"):
  par_err = err.RandomReservoir(network_name = "Random Network: %s" % target_db_name)
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


# ---------------------------------------------- #
## Function to extract best parameters
def extract_parms(parm_df_name):
  parm_df1 = pd.read_csv('07_ESN_MemoryOut/%s_df1.csv' % parm_df_name)
  parm_df2 = pd.read_csv('07_ESN_MemoryOut/%s_df2.csv' % parm_df_name)
  parm_df3 = pd.read_csv('07_ESN_MemoryOut/%s_df3.csv' % parm_df_name)
  best_n_node = 1000 # fixed
  best_alpha = 0.99 # fixed
  out_temp = parm_df2.loc[parm_df2["alpha"].astype("float") == best_alpha,]; out_temp = out_temp.reset_index(drop = True)
  best_leak_rate = float(out_temp.iloc[out_temp["NMSE_test"].astype("float64").idxmin(),].loc["leak_rate"])
  best_w_in_sparsity = float(parm_df3.iloc[parm_df3["NMSE_test"].astype("float64").idxmin(),].loc["Win_sparsity"])
  best_w_in_strength = float(parm_df3.iloc[parm_df3["NMSE_test"].astype("float64").idxmin(),].loc["Win_strength"])
  best_w_sparsity = float(parm_df3.iloc[parm_df3["NMSE_test"].astype("float64").idxmin(),].loc["W_sparsity"])
  # Return parameters
  return [best_n_node, best_alpha, best_leak_rate, best_w_in_sparsity, best_w_in_strength, best_w_sparsity]
# ---------------------------------------------- #


# Set target variable
## Collect parameters
pm = extract_parms("ESN_runif_ts")
## Prepare output data.frame
memory_out = pd.DataFrame()
for delay_step in np.arange(1,151,5):
  target_var, target_ts, target_db_name = 'random_01', runif_ts.iloc[delay_step:,:], "runif_ts"
  true_var, true_ts = 'random_01', runif_ts.iloc[:-delay_step,:]
  test_fraction = 0.5
  memory_tmp = reservoir_computing_parallel(pm[0], pm[1], target_ts, target_var, true_ts, true_var,
                                            w_in_sparsity=pm[3], w_in_strength=pm[4], w_sparsity=pm[5],
                                            leak_rate=pm[2], return_obj="class")
  memory_tmp.result_summary_df["delay"] = delay_step
  memory_out = memory_out.append(memory_tmp.result_summary_df)
memory_out = memory_out.reset_index(drop = True)
memory_out.to_csv('%s/ESN_Memory_df.csv' % (output_dir))

# Calculate memory capacity
mc_df_all = pd.DataFrame()
for num_nodes in np.arange(1, 1001, 100):
  memory_out2 = pd.DataFrame()
  for alpha in np.arange(0.1,1.3,0.2):
    for delay_step in np.arange(1,151,15):
      target_var, target_ts, target_db_name = 'random_01', runif_ts.iloc[delay_step:,:], "runif_ts"
      true_var, true_ts = 'random_01', runif_ts.iloc[:-delay_step,:]
      test_fraction = 0.5
      memory_tmp = reservoir_computing_parallel(num_nodes, alpha, target_ts, target_var, true_ts, true_var,
                                                w_in_sparsity=pm[3], w_in_strength=pm[4], w_sparsity=pm[5],
                                                leak_rate=pm[2], return_obj="class")
      memory_tmp.result_summary_df["delay"] = delay_step
      memory_out2 = memory_out2.append(memory_tmp.result_summary_df)
    # Calculate memory capacity
    esn_mc = sum(memory_out2["test_pred"].astype("float")**2)
    mc_df_tmp = memory_tmp.result_summary_df
    mc_df_tmp["memory_capacity"] = round(esn_mc, 3)
    mc_df_all = mc_df_all.append(mc_df_tmp)

mc_df_all = mc_df_all.reset_index(drop = True)
mc_df_all.to_csv("%s/ESN_MemoryCapacity.csv" % output_dir)

# Extract reservoir states with random inputs
esn_random_input = reservoir_computing_parallel(pm[0], pm[1], target_ts, target_var, true_ts, true_var,
                                                w_in_sparsity=pm[3], w_in_strength=pm[4], w_sparsity=pm[5],
                                                leak_rate=pm[2], return_obj="class")
esn_random_input_df = pd.DataFrame(esn_random_input.record_reservoir_nodes)
esn_random_input_df.to_csv("%s/ESN_RandomInput_ReservoirState.csv" % output_dir)





# ------------------------------------------------------------------------------------------ #
# ------------------------------------- Parameter search ----------------------------------- #
# ------------------------------------------------------------------------------------------ #

# Set tp = -1 as a base time point to measure the memory capacity
delay_step = 10
target_var, target_ts, target_db_name = 'random_01', runif_ts.iloc[delay_step:,:], "runif_ts"
true_var, true_ts = 'random_01', runif_ts.iloc[:-delay_step,:]

# The range of number of nodes
node_range = [[1], np.arange(20,1001,20).tolist()]
node_range = sum(node_range, [])
alpha_range = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1, 1.01, 1.05, 1.1, 1.3, 1.5]
Leak_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
## For grid search
WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WinSt_range = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
WSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Multiprocess computing
par_rc1 = joblib.Parallel(n_jobs=-1)([joblib.delayed(reservoir_computing_parallel)
                                     (x, 0.99, target_ts, target_var, true_ts, true_var, leak_rate = 0) for x in node_range])
par_rc2 = joblib.Parallel(n_jobs=-1, verbose = 10)([joblib.delayed(reservoir_computing_parallel)
                         (1000, x, target_ts, target_var, true_ts, true_var, leak_rate = y)
                         for x, y in itertools.product(alpha_range, Leak_range)])
# Compile output
output_all_df1 = par_rc1[0]
for i in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[i])
output_all_df2 = par_rc2[0]
for i in range(1,len(par_rc2)): output_all_df2 = output_all_df2.append(par_rc2[i])
# Reset indices
output_all_df1 = output_all_df1.reset_index(drop = True); output_all_df1.to_csv('%s/ESN_%s_df1.csv' % (output_dir, target_db_name))
output_all_df2 = output_all_df2.reset_index(drop = True); output_all_df2.to_csv('%s/ESN_%s_df2.csv' % (output_dir, target_db_name))
# Check values
best_n_node = 1000
best_alpha = 0.99
out_temp = output_all_df2.loc[output_all_df2["alpha"].astype("float") == best_alpha,]; out_temp = out_temp.reset_index(drop = True)
best_leak_rate = float(out_temp.iloc[out_temp["NMSE_test"].astype("float64").idxmin(),].loc["leak_rate"])
# w_in_sparsity, w_in_strength, w_sparsity
par_rc3 = joblib.Parallel(n_jobs=-1, verbose = 10)([joblib.delayed(reservoir_computing_parallel)
                         (best_n_node, best_alpha, target_ts, target_var, true_ts, true_var, w_in_sparsity = x, w_in_strength = y, w_sparsity = z, leak_rate = best_leak_rate)
                         for x, y, z in itertools.product(WinSp_range, WinSt_range, WSp_range)])
output_all_df3 = par_rc3[0]
for i in range(1,len(par_rc3)): output_all_df3 = output_all_df3.append(par_rc3[i])
# Reset indices
output_all_df3 = output_all_df3.reset_index(drop = True); output_all_df3.to_csv('%s/ESN_%s_df3.csv' % (output_dir, target_db_name))


# Extract parameters
pm = extract_parms("ESN_runif_ts")

# -------------------------------------------------- #
# Check single best Res
# -------------------------------------------------- #
# Single best random reservoir
rrc = err.RandomReservoir(network_name = "Random Network")
rrc.prepare_target_data(target_ts, target_var, true_ts, true_var, test_fraction = 0.5)
rrc.initialize_reservoir(num_reservoir_nodes = pm[0], alpha = pm[1],
                         w_in_sparsity = pm[3], w_in_strength = pm[4],
                         w_sparsity = pm[5], leak_rate = pm[2])
rrc.compute_reservoir_state()
rrc.learn_model(ridge_lambda = 0.05, washout_fraction = 0.05)
rrc.predict()
rrc.summarize_stat()
rrc.result_summary_df

# Save result to pickel
joblib.dump(rrc, "%s/ESN_Eml%s.jb" % (output_dir, target_db_name), compress=3)
# -------------------------------------------------- #
# -------------------------------------------------- #


