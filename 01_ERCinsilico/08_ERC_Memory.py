####
#### Ecological Reservoir Computing
#### No.8 Measuring memory capacity of ecological reservoir
####

# Import essential modules
import numpy as np
import pandas as pd
import itertools; import joblib; import time
from scipy import linalg
import module.helper_func_v20200617 as helper

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import custom module
import module.emulate_by_ecological_reservoir_v1 as erc
import module.emulate_by_multiple_reservoir_v1 as msr

# Create output directory
import os; output_dir = "08_ERC_MemoryOut"; os.mkdir(output_dir)

# -------------------- Prepare data and set parameters -------------------- #
# Load MNIST image data downloading
ecol_ts = pd.read_csv('./data/edna_asv_table_prok_all.csv')
#fish_ts = pd.read_csv('./data/BCreation_s0001_weekly.csv')
runif_ts = pd.read_csv("./data/runif_ts.csv").iloc[6000:7000,]
best_e = pd.read_csv("./data/bestE_all.csv")


# Define function for parallel computing
# ------------------------------------------------------------------------------------------ #
def single_simplex_reservoir(reservoir_var, reservoir_ts, reservoir_db_name,
                                 target_var, target_ts, target_db_name,
                                 true_var, true_ts, test_fraction = 0.5, n_nodes = None,
                                 w_in_strength = 0.5, w_in_sparsity = 0, w_in_seed = 1234,
                                 n_nn = None, leak = 0, bestE_data = best_e, return_obj = "summary"):
  if n_nodes == None:
    E = int(bestE_data[bestE_data["variable"] == reservoir_var]["bestE"])
  else:
    E = n_nodes
  sgl_src = erc.SimplexReservoir(reservoir_var, reservoir_ts, reservoir_db_name)
  sgl_src.compile_reservoir_data(E)
  sgl_src.prepare_target_data(target_ts, target_var, target_db_name, true_ts, true_var, test_fraction = test_fraction)
  sgl_src.initialize_reservoir(w_in_strength = w_in_strength, w_in_sparsity = w_in_sparsity, w_in_seed = w_in_seed, n_nn = n_nn, leak_rate = leak)
  sgl_src.compute_reservoir_state(initial_method = "zero")
  sgl_src.learn_model(ridge_lambda = 0.1)
  sgl_src.predict()
  sgl_src.summarize_stat()
  if return_obj == "summary":
    return sgl_src.result_summary_df
  else:
    return sgl_src
# ------------------------------------------------------------------------------------------ #

# Define function for parallel computing
# ------------------------------------------------------------------------------------------ #
def memory_cap_parallel(reservoir_var, reservoir_ts, reservoir_db_name,
                                 target_var, true_var, target_db_name,
                                 target_ts_original = None,
                                 true_ts_original = None,
                                 delay_step = 0,
                                 test_fraction = 0.5,
                                 w_in_strength = 0.5, w_in_sparsity = 0, w_in_seed = 1234,
                                 leak = 0, bestE_data = best_e):
  target_ts = target_ts_original.iloc[delay_step:,:]
  true_ts = true_ts_original.iloc[:-delay_step,:]
  memory_out_df = single_simplex_reservoir(reservoir_var, reservoir_ts, reservoir_db_name,
                                   target_var, target_ts, target_db_name, true_var, true_ts,
                                   test_fraction = test_fraction,
                                   n_nodes = None,
                                   w_in_strength = w_in_strength,
                                   w_in_sparsity = w_in_sparsity,
                                   w_in_seed = w_in_seed,
                                   leak = leak, 
                                   n_nn = None, bestE_data = best_e, return_obj = "summary")
  memory_out_df["delay"] = delay_step
  return memory_out_df
# ------------------------------------------------------------------------------------------ #



# !!--- Load after the parameter search ---!!
parms_df = pd.read_csv('%s/best_parms_runif_ts_df.csv' % (output_dir))
# Set target variable
reservoir_var, reservoir_ts, reservoir_db_name = "Prok_Taxa00004", ecol_ts, "ecol_ts"

# ----------------------- For all species with single delay step -------------------------- #
# Test memory capacity at delay = 2
delay_step = 2
memory_out1 = pd.DataFrame()
n_spp = 500
# Start analysis
ecol_ts_colnames = ecol_ts.iloc[:,1:].columns
reservoir_var_list = ecol_ts_colnames[0:n_spp]
delay_step_i = 5
sp_opt_var = pd.Index(parms_df["w_in_sp_opt"])
st_opt_var = pd.Index(parms_df["w_in_st_opt"])
leak_opt_var = pd.Index(parms_df["leak_opt"])
# Specify data set
target_var, true_var, target_db_name = 'random_01', 'random_01', "runif_ts"
# Parallel computing
par_memory = joblib.Parallel(n_jobs=-1)([joblib.delayed(memory_cap_parallel)
                               (x, reservoir_ts, reservoir_db_name,
                               target_var, true_var, target_db_name,
                               target_ts_original = runif_ts,
                               true_ts_original = runif_ts,
                               test_fraction = 0.5,
                               delay_step = delay_step_i,
                               w_in_sparsity = y,
                               w_in_strength = z, leak = l,
                               bestE_data = best_e)
                               for x, y, z, l in zip(reservoir_var_list, sp_opt_var, st_opt_var, leak_opt_var)])
memory_out1 = par_memory[0]
for i in range(1,len(par_memory)): memory_out1 = memory_out1.append(par_memory[i])
memory_out1 = memory_out1.reset_index(drop = True)
memory_out1.to_csv('%s/ERC_Memory_AllSp_Delay%s.csv' % (output_dir, delay_step_i))
# ------------------------------------------------------------------------------------------ #


# ----------------------- For all species with multiple delay steps -------------------------- #
# Set target variable
reservoir_var, reservoir_ts, reservoir_db_name = "Prok_Taxa00004", ecol_ts, "ecol_ts"
# Test memory capacity at delay = 2
#delay_step = 2
memory_out2 = pd.DataFrame()
n_spp = 500
# Start analysis
ecol_ts_colnames = ecol_ts.iloc[:,1:].columns
reservoir_var_list = ecol_ts_colnames[0:n_spp]
delay_step_var = np.arange(1,21,1)
sp_opt_var = pd.Index(parms_df["w_in_sp_opt"])
st_opt_var = pd.Index(parms_df["w_in_st_opt"])
leak_opt_var = pd.Index(parms_df["leak_opt"])
# Specify data set
target_var, true_var, target_db_name = 'random_01', 'random_01', "runif_ts"
mc_df_all = pd.DataFrame()
for m in delay_step_var:
  # Parallel computing
  par_memory_all = joblib.Parallel(n_jobs=-1)([joblib.delayed(memory_cap_parallel)
                                 (x, reservoir_ts, reservoir_db_name,
                                 target_var, true_var, target_db_name,
                                 target_ts_original = runif_ts,
                                 true_ts_original = runif_ts,
                                 test_fraction = 0.5,
                                 delay_step = m,
                                 w_in_sparsity = y,
                                 w_in_strength = z, leak = l,
                                 bestE_data = best_e)
                                 for x, y, z, l in zip(reservoir_var_list, sp_opt_var, st_opt_var, leak_opt_var)])
  mc_df_tmp = par_memory_all[0]
  for j in range(1,len(par_memory_all)): mc_df_tmp = mc_df_tmp.append(par_memory_all[j])
  mc_df_all = mc_df_all.append(mc_df_tmp)
  print('Delay step %s finished' % m)
mc_df_all = mc_df_all.reset_index(drop = True)
mc_df_all.to_csv('%s/ERC_Memory_AllSp_AllDelay.csv' % output_dir)
# ------------------------------------------------------------------------------------------ #
mc_df_all = pd.read_csv('%s/ERC_Memory_AllSp_AllDelay.csv' % output_dir)
mc_df_all.loc[mc_df_all["reservoir_var"]=="Prok_Taxa00004"]
mc_df_all.loc[mc_df_all["test_pred"].astype("float") < 0, "test_pred"] = 0
mc_df_all["r2"] = mc_df_all["test_pred"].astype("float")**2
mc_df_all.to_csv('%s/ERC_Memory_AllSp_AllDelayMemory.csv' % output_dir)


# Calculating "Forgatten curve" for the best species
mc_df_all.groupby('reservoir_var').mean().idxmax()['r2']
mc_df_all[mc_df_all["reservoir_var"]=="Prok_Taxa00207"]
best_var = mc_df_all.groupby('reservoir_var').mean().idxmax()['r2']
bpm = parms_df[parms_df["Taxa"]==best_var]

memory_out2 = pd.DataFrame()
for delay_step in np.arange(1,51,1):
  target_var, target_ts, target_db_name = 'random_01', runif_ts.iloc[delay_step:,:], "runif_ts"
  true_var, true_ts = 'random_01', runif_ts.iloc[:-delay_step,:]
  memory_tmp = single_simplex_reservoir(best_var, reservoir_ts, reservoir_db_name,
                                 target_var, target_ts, target_db_name,
                                 true_var, true_ts,
                                 w_in_strength = float(bpm["w_in_st_opt"]),
                                 w_in_sparsity = float(bpm["w_in_sp_opt"]),
                                 leak = float(bpm["leak_opt"]),
                                 return_obj="class")
  memory_tmp.result_summary_df["delay"] = delay_step
  memory_out2 = memory_out2.append(memory_tmp.result_summary_df)
memory_out2 = memory_out2.reset_index(drop = True)
memory_out2.to_csv('%s/ERC_BestTaxa_Memory_df.csv' % (output_dir))


# ------------------------------------------------------------------------------------------ #
# --------------------------------- Multi-network memory ----------------------------------- #
# ------------------------------------------------------------------------------------------ #
# Prepare output dataframe
multinetwork_df = pd.DataFrame()
for delay_step in np.arange(1,51,1):
  # Select data
  reservoir_ts, reservoir_db_name = ecol_ts, "ecol_ts"
  target_var, target_ts, target_db_name = 'random_01', runif_ts.iloc[delay_step:,:], "runif_ts"
  true_var, true_ts = 'random_01', runif_ts.iloc[:-delay_step,:]
  for n_spp in [500]:
    # Select variables
    ecol_ts_colnames = ecol_ts.iloc[:,1:].columns
    reservoir_var_list = ecol_ts_colnames[0:n_spp]
    sp_opt_var = pd.Index(parms_df["w_in_sp_opt"])
    st_opt_var = pd.Index(parms_df["w_in_st_opt"])
    leak_opt_var = pd.Index(parms_df["leak_opt"])
    # Multiprocess computing
    par_erc = joblib.Parallel(n_jobs=-1)([joblib.delayed(single_simplex_reservoir)
                                        (x, reservoir_ts, reservoir_db_name,
                                         target_var, target_ts, target_db_name,
                                         true_var, true_ts, test_fraction = 0.5,
                                         w_in_sparsity = y, w_in_strength = z, 
                                         leak = l, n_nodes = None,
                                         bestE_data = best_e, return_obj = "class")
                                          for x, y, z, l in zip(reservoir_var_list, sp_opt_var, st_opt_var, leak_opt_var)])
    output_erc = par_erc[0].result_summary_df
    for i in range(1,len(par_erc)): output_erc = output_erc.append(par_erc[i].result_summary_df)
    
    n_network = output_erc.shape[0] - 1
    # Combine reservoir states ----------------------------------------------------------------- #
    combined_reservoir_state = []; combined_test_reservoir_state = [] # Initialize
    total_E = 0
    if n_network == 0:
          combined_reservoir_state = par_erc[0].record_reservoir_nodes
          combined_test_reservoir_state = par_erc[0].test_reservoir_nodes
          total_E = total_E + par_erc[0].num_nodes
    else:
      for sp_i in range(n_network + 1):
        if len(combined_reservoir_state) == 0:
          combined_reservoir_state = par_erc[sp_i].record_reservoir_nodes
          combined_test_reservoir_state = par_erc[sp_i].test_reservoir_nodes
          total_E = total_E + par_erc[sp_i].num_nodes
        else:
          combined_reservoir_state = np.hstack([combined_reservoir_state, par_erc[sp_i].record_reservoir_nodes])
          combined_test_reservoir_state = np.hstack([combined_test_reservoir_state, par_erc[sp_i].test_reservoir_nodes])
          total_E = total_E + par_erc[sp_i].num_nodes
      # ------------------------------------------------------------------------------------------ #
    
    # Perform multiNetwork analysis
    emr01 = msr.MultinetworkSimplexReservoir("Combined_Network_%s" % target_db_name)
    emr01.learn_model(combined_reservoir_state, par_erc[0].train_true, ridge_lambda = 1)
    emr01.predict(combined_test_reservoir_state, par_erc[0].test_true)
    emr01.summarize_stat()
    emr01.result_summary_df["n_network"] = n_network + 1
    emr01.result_summary_df["E"] = total_E
    emr01.result_summary_df["delay"] = delay_step
    emr01.result_summary_df = pd.concat([emr01.result_summary_df, par_erc[0].result_summary_df.iloc[:,10:15]], axis = 1)
    multinetwork_df = multinetwork_df.append(emr01.result_summary_df)
  
multinetwork_df = multinetwork_df.reset_index(drop = True)
multinetwork_df.to_csv("%s/MultiERC_Memory_%s.csv" % (output_dir, target_db_name))
combined_reservoir_state_df = pd.DataFrame(combined_reservoir_state)
combined_reservoir_state_df.to_csv("%s/MultiERC_ReservoirState_%s.csv" % (output_dir, target_db_name))
joblib.dump(emr01, "%s/MultiERC_%s.jb" % (output_dir, target_db_name), compress=3)
# ------------------------------------------------------------------------------------------ #





# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------- Parameter search ----------------------------------- #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# Set target variables
delay_step=5
reservoir_var, reservoir_ts, reservoir_db_name = "Prok_Taxa00004", ecol_ts, "ecol_ts"
target_var, target_ts, target_db_name = 'random_01', runif_ts.iloc[delay_step:,:], "runif_ts"
true_var, true_ts = 'random_01', runif_ts.iloc[:-delay_step,:]

# Search parameters
node_range = [[2,3,4,5,6,7,8,9,10,15], np.arange(20,101,10).tolist()]
node_range = sum(node_range, [])

# Multiprocess computing
par_rc1 = joblib.Parallel(n_jobs=1)([joblib.delayed(single_simplex_reservoir)
                                    ("Prok_Taxa00004", reservoir_ts, reservoir_db_name,
                                     target_var, target_ts, target_db_name,
                                     true_var, true_ts, n_nodes = x) for x in node_range])
# Compile output
output_all_df1 = par_rc1[0]
for i in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[i])
# Reset indices
output_all_df1 = output_all_df1.reset_index(drop = True); output_all_df1.to_csv('%s/ERC_%s_df1.csv' % (output_dir, target_db_name))

# Search parameters
Leak_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WinSt_range = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]

# Search parameters
# Grid search for each species
ecol_ts_colnames = ecol_ts.iloc[:,1:].columns
parms_df = pd.DataFrame(columns=['Taxa','w_in_sp_opt','w_in_st_opt','leak_opt'])
for i in range(ecol_ts_colnames.shape[0]):
  parms_df.loc[i] = [ecol_ts_colnames[i], 0, 0, 0]

for i in range(ecol_ts_colnames.shape[0]):
  tax_i = parms_df["Taxa"][i]
  par_rc = joblib.Parallel(n_jobs = -1, verbose=10)([joblib.delayed(single_simplex_reservoir)
                                      (tax_i, reservoir_ts, reservoir_db_name,
                                       target_var, target_ts, target_db_name,
                                       true_var, true_ts, test_fraction = 0.5,
                                       w_in_sparsity = x, w_in_strength = y, leak = z)
                           for x, y, z in itertools.product(WinSp_range, WinSt_range, Leak_range)])
  output_all_df = par_rc[0]
  for j in range(1,len(par_rc)): output_all_df = output_all_df.append(par_rc[j]);
  output_all_df = output_all_df.reset_index(drop = True)
  w_in_sp_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"Win_sparsity"]
  w_in_st_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"Win_strength"]
  leak_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"leak_rate"]
  parms_df.loc[i]["w_in_sp_opt"] = float(np.array(w_in_sp_min)[0])
  parms_df.loc[i]["w_in_st_opt"] = float(np.array(w_in_st_min)[0])
  parms_df.loc[i]["leak_opt"] = float(np.array(leak_min)[0])

# Save best parameters
parms_df = parms_df.reset_index(drop = True)
parms_df.to_csv('%s/best_parms_%s_df.csv' % (output_dir, target_db_name))

# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
