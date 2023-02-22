####
#### Ecological Reservoir Computing
#### No.4 Simplex reservoir using ecological time series
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
import module.predict_by_ecological_reservoir_v1 as erc
import module.predict_by_multiple_reservoir as msr

# Create output directory
import os; output_dir = "04_ERC_LorenzPredOut"; os.mkdir(output_dir)

# -------------------- Prepare data and set parameters -------------------- #
# Load MNIST image data downloading
lorenz = pd.read_csv("./data/LorenzTS_tau10.csv")
#logis_ts = pd.read_csv("./data/logistic_Xr2_92Yr2_90.csv")
ecol_ts = pd.read_csv("./data/edna_asv_table_prok_all.csv")
#fish_ts = pd.read_csv("./data/BCreation_s0001_weekly.csv")
best_e = pd.read_csv("./data/bestE_all.csv")
# !!--- Load after the parameter search ---!!
parms_df = pd.read_csv("04_ERC_LorenzPredOut/best_parms_df.csv")

# Target vars
target_var, target_ts = 'x', lorenz[91:]

# Define wrapper function
# ------------------------------------------------------------------------------------------ #
def single_simplex_reservoir(reservoir_var, reservoir_ts, reservoir_db_name,
                                 target_var, target_ts, target_db_name, test_fraction = 0.2,
                                 n_nodes = None, w_in_strength = 2, w_in_sparsity = 0, n_nn = None,
                                 leak = 0, return_obj = "summary", bestE_data = best_e):
  if n_nodes == None:
    E = int(bestE_data[bestE_data["variable"] == reservoir_var]["bestE"])
  else:
    E = n_nodes
  sgl_src = erc.SimplexReservoir(reservoir_var, reservoir_ts, reservoir_db_name)
  sgl_src.compile_reservoir_data(E)
  sgl_src.prepare_target_data(target_ts, target_var, target_db_name, test_fraction)
  sgl_src.initialize_reservoir(w_in_strength = w_in_strength, w_in_sparsity = w_in_sparsity, n_nn = n_nn, leak_rate = leak)
  sgl_src.compute_reservoir_state(initial_method = "zero")
  sgl_src.learn_model(ridge_lambda = 1)
  sgl_src.predict()
  sgl_src.summarize_stat()
  if return_obj == "summary":
    return sgl_src.result_summary_df
  else:
    return sgl_src
# ------------------------------------------------------------------------------------------ #


# Perform multinetwork approach ------------------------------------------------------------------- #
for n_spp in [500]:
  # Select variables
  ecol_ts_colnames = ecol_ts.iloc[:,1:].columns
  reservoir_var_list = ecol_ts_colnames[0:n_spp]
  sp_opt_var = pd.Index(parms_df["w_in_sp_opt"])
  st_opt_var = pd.Index(parms_df["w_in_st_opt"])
  # Multiprocess computing
  par_erc = joblib.Parallel(n_jobs=-1)([joblib.delayed(single_simplex_reservoir)
                                      (x, reservoir_ts, reservoir_db_name,
                                       target_var, target_ts, target_db_name,
                                       w_in_sparsity = y, w_in_strength = z, 
                                       bestE_data = best_e, return_obj = "class")
                                       for x, y, z in zip(reservoir_var_list, sp_opt_var, st_opt_var)])
  
  output_erc = par_erc[0].result_summary_df
  for i in range(1,len(par_erc)): output_erc = output_erc.append(par_erc[i].result_summary_df)
  # Check maximum prediction skill when single species reservoir is used --------------------- #
  # Prepare output dataframe
  multinetwork_df = pd.DataFrame()
  
  # Choose n_network range
  for n_network in range(0, output_erc.shape[0]):
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
    msr01 = msr.MultinetworkSimplexReservoir("Combined_Network")
    msr01.learn_model(combined_reservoir_state, par_erc[0].train_true, ridge_lambda = 1)
    msr01.predict(combined_test_reservoir_state, par_erc[0].test_true)
    msr01.summarize_stat()
    msr01.result_summary_df["n_network"] = n_network + 1
    msr01.result_summary_df["E"] = total_E
    msr01.result_summary_df = pd.concat([msr01.result_summary_df, par_erc[0].result_summary_df.iloc[:,9:14]], axis = 1)
    multinetwork_df = multinetwork_df.append(msr01.result_summary_df)
  
  multinetwork_df = multinetwork_df.reset_index(drop = True)
  multinetwork_df.to_csv("%s/MultiERC_bestE.csv" % output_dir)

# ------------------------------------------------------------------------------------------ #
# Save output
joblib.dump(msr01, "%s/ERC_MultiPredLorenz.jb" % output_dir, compress=3)



# ------------------------------------------------------------------------------------------ #
# -----------------------------      Parameter search     ---------------------------------- #
# ------------------------------------------------------------------------------------------ #
# Perform ecological reservoir computing using optimal E
reservoir_ts, reservoir_db_name = ecol_ts, "ecol_ts"
target_var, target_ts, target_db_name = "x", lorenz[91:], "lorenz"
var_all = ecol_ts.columns[1:]
single_erc_res = pd.DataFrame()
for reservoir_var in var_all:
  erc_res = single_simplex_reservoir(reservoir_var, reservoir_ts, reservoir_db_name,
                                     target_var, target_ts, target_db_name, bestE_data = best_e)
  single_erc_res = single_erc_res.append(erc_res)
## Save results
single_erc_res = single_erc_res.reset_index(drop = True)
single_erc_res.to_csv("%s/SingleERC_bestE.csv" % output_dir)
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# -----------------------------      Parameter search     ---------------------------------- #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# Evaluating performce by grid search
# Set target variables
reservoir_var, reservoir_ts, reservoir_db_name = "Prok_Taxa00006", ecol_ts, "ecol_ts"
target_var, target_ts, target_db_name = "x", lorenz[91:], "lorenz"

# Search parameters
node_range = [[2,3,4,5,6,7,8,9,10,15], np.arange(20,101,10).tolist()]
node_range = sum(node_range, [])

# Multiprocess computing
par_rc1 = joblib.Parallel(n_jobs=1)([joblib.delayed(single_simplex_reservoir)
                                    ("Prok_Taxa00004", reservoir_ts, reservoir_db_name,
                                     target_var, target_ts, target_db_name, n_nodes = x) for x in node_range])
par_rc2 = joblib.Parallel(n_jobs=1)([joblib.delayed(single_simplex_reservoir)
                                    ("Prok_Taxa00005", reservoir_ts, reservoir_db_name,
                                     target_var, target_ts, target_db_name, n_nodes = x) for x in node_range])
par_rc3 = joblib.Parallel(n_jobs=1)([joblib.delayed(single_simplex_reservoir)
                                    ("Prok_Taxa00006", reservoir_ts, reservoir_db_name, 
                                     target_var, target_ts, target_db_name, n_nodes = x) for x in node_range])
# Compile output
output_all_df1 = par_rc1[0]
for i in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[i])
output_all_df2 = par_rc2[0]
for i in range(1,len(par_rc2)): output_all_df2 = output_all_df2.append(par_rc2[i])
output_all_df3 = par_rc3[0]
for i in range(1,len(par_rc3)): output_all_df3 = output_all_df3.append(par_rc3[i])
# Reset indices
output_all_df1 = output_all_df1.reset_index(drop = True); output_all_df1.to_csv('%s/ERC_output_df1.csv' % output_dir)
output_all_df2 = output_all_df2.reset_index(drop = True); output_all_df2.to_csv('%s/ERC_output_df2.csv' % output_dir)
output_all_df3 = output_all_df3.reset_index(drop = True); output_all_df3.to_csv('%s/ERC_output_df3.csv' % output_dir)

# Search parameters
WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WinSt_range = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

# Grid search for each species
ecol_ts_colnames = ecol_ts.iloc[:,1:].columns
parms_df = pd.DataFrame(columns=['Taxa','w_in_sp_opt','w_in_st_opt'])
for i in range(ecol_ts_colnames.shape[0]):
  parms_df.loc[i] = [ecol_ts_colnames[i], 0, 0]

for i in range(ecol_ts_colnames.shape[0]):
  tax_i = parms_df["Taxa"][i]
  par_rc4 = joblib.Parallel(n_jobs=-1)([joblib.delayed(single_simplex_reservoir)
                                      (tax_i, reservoir_ts, reservoir_db_name,
                                       target_var, target_ts, target_db_name,
                                       bestE_data = best_e,
                                       w_in_sparsity = x, w_in_strength = y)
                                       for x, y in itertools.product(WinSp_range, WinSt_range)])
  output_all_df4 = par_rc4[0]
  for j in range(1,len(par_rc4)):
    output_all_df4 = output_all_df4.append(par_rc4[j])
  output_all_df4 = output_all_df4.reset_index(drop = True)
  w_in_sp_min = output_all_df4.loc[output_all_df4['NMSE_test'].min() == output_all_df4['NMSE_test'],"Win_sparsity"]
  w_in_st_min = output_all_df4.loc[output_all_df4['NMSE_test'].min() == output_all_df4['NMSE_test'],"Win_strength"]
  parms_df.loc[i]["w_in_sp_opt"] = np.array(w_in_sp_min)[0]
  parms_df.loc[i]["w_in_st_opt"] = np.array(w_in_st_min)[0]

# Save best parameters
parms_df = parms_df.reset_index(drop = True); parms_df.to_csv('%s/best_parms_df.csv' % output_dir)
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
