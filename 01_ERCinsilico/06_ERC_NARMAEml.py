####
#### Ecological Reservoir Computing with python
#### No.6 Emulate NARMA by ERC
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
import os; output_dir = "06_ERC_NARMAEmlOut"; os.mkdir(output_dir)

# -------------------- Prepare data and set parameters -------------------- #
# Load MNIST image data downloading
ecol_ts = pd.read_csv('./data/edna_asv_table_prok_all.csv')
#fish_ts = pd.read_csv('./data/BCreation_s0001_weekly.csv')
best_e = pd.read_csv("./data/bestE_all.csv")
# Load NARMA time series
narma02 = pd.read_csv('./data/NARMA02.csv').iloc[6000:7000,]
narma03 = pd.read_csv('./data/NARMA03.csv').iloc[6000:7000,]
narma04 = pd.read_csv('./data/NARMA04.csv').iloc[6000:7000,]
narma05 = pd.read_csv('./data/NARMA05.csv').iloc[6000:7000,]
narma10 = pd.read_csv('./data/NARMA10.csv').iloc[6000:7000,]

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

# Set target variables
reservoir_var, reservoir_ts, reservoir_db_name = "Prok_Taxa00004", ecol_ts, "ecol_ts"
# target_var, target_ts, target_db_name = 'input', narma02.iloc[:-1,:], "narma02"
# true_var, true_ts = 'value', narma02.iloc[1:,:]
# target_var, target_ts, target_db_name = 'input', narma03.iloc[:-1,:], "narma03"
# true_var, true_ts = 'value', narma03.iloc[1:,:]
# target_var, target_ts, target_db_name = 'input', narma04.iloc[:-1,:], "narma04"
# true_var, true_ts = 'value', narma04.iloc[1:,:]
# target_var, target_ts, target_db_name = 'input', narma05.iloc[:-1,:], "narma05"
# true_var, true_ts = 'value', narma05.iloc[1:,:]
target_var, target_ts, target_db_name = 'input', narma10.iloc[:-1,:], "narma10"
true_var, true_ts = 'value', narma10.iloc[1:,:]


# !!--- Load after the parameter search ---!!
parms_df = pd.read_csv('%s/best_parms_%s_df.csv' % (output_dir, target_db_name))
# Perform multinetwork approach ------------------------------------------------------------------- #
for n_spp in [500]:
  step = 10
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
  # Save single-species ERC results
  output_erc.to_csv("%s/SingleERC_%s.csv" % (output_dir, target_db_name))
  
  # Prepare output dataframe
  multinetwork_df = pd.DataFrame()
  for n_network in np.append(np.arange(0, output_erc.shape[0], step), n_spp-1):
    # Combine reservoir states ----------------------------------------------------------------- #
    combined_reservoir_state = []; combined_test_reservoir_state = [] # Initialize
    total_E = 0
    if n_network == 0:
          combined_reservoir_state = par_erc[0].record_reservoir_nodes
          combined_test_reservoir_state = par_erc[0].test_reservoir_nodes
          total_E = total_E + par_erc[0].num_nodes
    else:
      for sp_i in range(n_network+1):
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
    emr01.result_summary_df = pd.concat([emr01.result_summary_df, par_erc[0].result_summary_df.iloc[:,10:15]], axis = 1)
    multinetwork_df = multinetwork_df.append(emr01.result_summary_df)
  multinetwork_df = multinetwork_df.reset_index(drop = True)
  multinetwork_df.to_csv("%s/MultiERC_%s.csv" % (output_dir, target_db_name))
multinetwork_df.iloc[:,0:5]
# ------------------------------------------------------------------------------------------ #
joblib.dump(emr01, "%s/MultiERC_%s.jb" % (output_dir, target_db_name), compress=3)




# ------------------------------------------------------------------------------------------ #
# -----------------------------      Parameter search     ---------------------------------- #
# ------------------------------------------------------------------------------------------ #
# Perform ecological reservoir computing using optimal E
reservoir_ts, reservoir_db_name = ecol_ts, "ecol_ts"
var_all = ecol_ts.columns[1:]
single_erc_res = pd.DataFrame()

for reservoir_var in var_all:
  erc_res = single_simplex_reservoir(reservoir_var, reservoir_ts, reservoir_db_name,
                                     target_var, target_ts, target_db_name,
                                     true_var, true_ts, bestE_data = best_e)
  single_erc_res = single_erc_res.append(erc_res)

single_erc_res = single_erc_res.reset_index(drop = True)
single_erc_res.to_csv("%s/SingleERC_%s.csv" % (output_dir, target_db_name))
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# -----------------------------      Parameter search     ---------------------------------- #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# Evaluating performce by grid search
# Set target variables
narma_names = ["narma02", "narma03", "narma04", "narma05", "narma10"]
for narma_i in narma_names:
  if narma_i == "narma02":
    target_var, target_ts, target_db_name = 'input', narma02.iloc[:-1,:], "narma02"
    true_var, true_ts = 'value', narma02.iloc[1:,:]
  if narma_i == "narma03":
    target_var, target_ts, target_db_name = 'input', narma03.iloc[:-1,:], "narma03"
    true_var, true_ts = 'value', narma03.iloc[1:,:]
  if narma_i == "narma04":
    target_var, target_ts, target_db_name = 'input', narma04.iloc[:-1,:], "narma04"
    true_var, true_ts = 'value', narma04.iloc[1:,:]
  if narma_i == "narma05":
    target_var, target_ts, target_db_name = 'input', narma05.iloc[:-1,:], "narma05"
    true_var, true_ts = 'value', narma05.iloc[1:,:]
  if narma_i == "narma10":
    target_var, target_ts, target_db_name = 'input', narma10.iloc[:-1,:], "narma10"
    true_var, true_ts = 'value', narma10.iloc[1:,:]
  
  # node_range
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
  output_all_df1 = output_all_df1.reset_index(drop = True)
  output_all_df1.to_csv('%s/ERC_%s_df1.csv' % (output_dir, target_db_name))
  
  # Search parameters
  #nn_range = [2, 3, 5, 10, 20, 30, 40, None]
  Leak_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  WinSt_range = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
  
  # Search parameters
  # Grid search for each species
  ecol_ts_colnames = ecol_ts.iloc[:,1:].columns
  parms_df = pd.DataFrame(columns=['Taxa','w_in_sp_opt','w_in_st_opt','leak_opt',
                                    'train_pred', 'test_pred', 'RMSE_train', 'RMSE_test', 'NMSE_train', 'NMSE_test'])
  others_col = ["train_pred", "test_pred", "RMSE_train", "RMSE_test", "NMSE_train", "NMSE_test"]
  for i in range(ecol_ts_colnames.shape[0]):
    parms_df.loc[i] = [ecol_ts_colnames[i], 0, 0, 0, 0, 0, 0, 0, 0, 0]
  
  for i in range(ecol_ts_colnames.shape[0]):
    tax_i = parms_df["Taxa"][i]
    par_rc = joblib.Parallel(n_jobs=-1, verbose = 10)([joblib.delayed(single_simplex_reservoir)
                                        (tax_i, reservoir_ts, reservoir_db_name,
                                         target_var, target_ts, target_db_name,
                                         true_var, true_ts,
                                         w_in_sparsity = x, w_in_strength = y,
                                         leak = z) for x, y, z in itertools.product(WinSp_range, WinSt_range, Leak_range)])
    output_all_df = par_rc[0]
    for j in range(1,len(par_rc)):
      output_all_df = output_all_df.append(par_rc[j])
    output_all_df = output_all_df.reset_index(drop = True)
    w_in_sp_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"Win_sparsity"]
    w_in_st_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"Win_strength"]
    leak_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"leak_rate"]
    others_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'], others_col]
    parms_df.loc[i]["w_in_sp_opt"] = float(np.array(w_in_sp_min)[0])
    parms_df.loc[i]["w_in_st_opt"] = float(np.array(w_in_st_min)[0])
    parms_df.loc[i]["leak_opt"] = float(np.array(leak_min)[0])
    parms_df.loc[i][others_col] = others_min.astype("float").values[0,:]
  
  # Save best parameters
  parms_df = parms_df.reset_index(drop = True)
  parms_df.to_csv('%s/best_parms_%s_df.csv' % (output_dir, target_db_name))
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
