####
#### Ecological Reservoir Computing
#### No.10 Emulate Mackey-Glass in closed-loop by ESN
####

# Import modules
import numpy as np
import pandas as pd
import itertools; import joblib; import time
from scipy import linalg
import module.helper_func_v20200617 as helper

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import custom module
import module.closedloop_by_ecological_reservoir_v1 as cloop
import module.closedloop_by_multiple_reservoir_v1 as msr

# Create output directory
import os; output_dir = "10_ERC_MackeyGlassOut"; os.mkdir(output_dir)

# -------------------- Prepare data and set parameters -------------------- #
# Load ecological reservoir time series
ecol_ts = pd.read_csv('./data/edna_asv_table_prok_all.csv')
#fish_ts = pd.read_csv('./data/BCreation_s0001_weekly.csv')
mackey_ts = pd.read_csv("./data/mackey_glass17.csv").iloc[5000:7000,]
best_e = pd.read_csv("./data/bestE_all.csv")

# Set target variable
reservoir_var, reservoir_ts, reservoir_db_name = "Prok_Taxa00004", ecol_ts, "ecol_ts"

# Standardize mackey_ts
train_data_size = 1600
mackey_std = (mackey_ts["value"] - mackey_ts["value"].min())/(mackey_ts["value"].max() - mackey_ts["value"].min())
train_true = mackey_std.iloc[0:train_data_size]
test_true = mackey_std.iloc[train_data_size:]
#train_data = None # use constant input
#test_data = None # use constant input
train_data = mackey_ts["random_input"].iloc[0:train_data_size] # Dummy, constant input is used in the function
test_data = mackey_ts["random_input"].iloc[train_data_size:] # Dummy, constant input is used in the function


# --------------------- Example ------------------------ #
import importlib; importlib.reload(cloop)
cloop_01 = cloop.SimplexReservoir("Mackey_Glass_closed_loop", "Prok_Taxa00014", reservoir_ts, reservoir_db_name)
cloop_01.compile_reservoir_data(5)
cloop_01.prepare_data(train_data, train_true, test_data, test_true,
                      train_var = "Mackey_Glass", test_var = "None")
cloop_01.initialize_reservoir(w_in_sparsity = 0, w_back_sparsity = 0,
                         w_in_strength = 0.1, w_back_strength = 1.19,
                         Win_seed = 1, Wback_seed = 3)
cloop_01.compute_reservoir_state(const_input = 0.2, C1 = 0.44, a1 = 0.9)
cloop_01.learn_model(ridge_lambda = 0.05, washout_fraction = 0.05)
cloop_01.predict()
cloop_01.summarize_stat()
cloop_01.result_summary_df


def cloop_simplex_reservoir(reservoir_var, reservoir_ts, reservoir_db_name,
                            train_data, train_true, test_data, test_true,
                            n_nodes = None, bestE_data = best_e,
                            w_in_sparsity = 0, w_back_sparsity = 0,
                            w_in_strength = 0.1, w_back_strength = 1,
                            Win_seed = 1234, Wback_seed = 1235,
                            return_obj = "summary"):
  if n_nodes == None:
    E = int(bestE_data[bestE_data["variable"] == reservoir_var]["bestE"])
  else:
    E = n_nodes
  cloop_00 = cloop.SimplexReservoir("Mackey_Glass_closed_loop", reservoir_var, reservoir_ts, reservoir_db_name)
  cloop_00.compile_reservoir_data(E)
  cloop_00.prepare_data(train_data, train_true, test_data, test_true,
                        train_var = "Mackey_Glass", test_var = "None")
  cloop_00.initialize_reservoir(w_in_sparsity = w_in_sparsity, w_back_sparsity = w_back_sparsity,
                           w_in_strength = w_in_strength, w_back_strength = w_back_strength,
                           Win_seed = Win_seed, Wback_seed = Wback_seed)
  cloop_00.compute_reservoir_state(const_input = 0.2, C1 = 0.44, a1 = 0.9)
  cloop_00.learn_model(ridge_lambda = 0.05, washout_fraction = 0)
  cloop_00.predict()
  cloop_00.summarize_stat()
  if return_obj == "summary":
    return cloop_00.result_summary_df
  else:
    return cloop_00
# ------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# Parameter search
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
if False: # Run this if the parmeter search is not done
  # The range of number of nodes
  WinSp_range = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
  WinSt_range = [0.5, 0.8, 1.0, 1.5, 2.0]
  # Back parameters
  WbSp_range =  [0, 0.2, 0.4, 0.6, 0.8, 0.9]
  WbSt_range = [0.5, 0.8, 1.0, 1.5, 2.0]
  len(WinSp_range)*len(WinSt_range)*len(WbSp_range)*len(WbSt_range)
  
  ecol_ts_colnames = ecol_ts.iloc[:,1:].columns
  parms_df = pd.DataFrame(columns=['Taxa','w_in_sp_opt','w_in_st_opt','w_bc_sp_opt', 'w_bc_st_opt',
                                    'train_pred', 'test_pred', 'RMSE_train', 'RMSE_test', 'NMSE_train', 'NMSE_test'])
  others_col = ["train_pred", "test_pred", "RMSE_train", "RMSE_test", "NMSE_train", "NMSE_test"]
  for i in range(ecol_ts_colnames.shape[0]):
    parms_df.loc[i] = [ecol_ts_colnames[i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  
  for i in range(ecol_ts_colnames.shape[0]):
    tax_i = parms_df["Taxa"][i]
    par_rc = joblib.Parallel(n_jobs=-1, verbose = 5)([joblib.delayed(cloop_simplex_reservoir)
              (tax_i, reservoir_ts, reservoir_db_name,
               train_data, train_true, test_data, test_true,
               w_in_sparsity = x, w_in_strength = y,
               w_back_sparsity = z, w_back_strength = l)
               for x, y, z, l in itertools.product(WinSp_range, WinSt_range, WbSp_range, WbSt_range)])
    output_all_df = par_rc[0]
    for j in range(1,len(par_rc)): output_all_df = output_all_df.append(par_rc[j])
    output_all_df = output_all_df.reset_index(drop = True)
    w_in_sp_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"Win_sparsity"]
    w_in_st_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"Win_strength"]
    w_bc_sp_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"Wback_sparsity"]
    w_bc_st_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'],"Wback_strength"]
    others_min = output_all_df.loc[output_all_df['NMSE_test'].min() == output_all_df['NMSE_test'], others_col]
    parms_df.loc[i]["w_in_sp_opt"] = float(np.array(w_in_sp_min)[0])
    parms_df.loc[i]["w_in_st_opt"] = float(np.array(w_in_st_min)[0])
    parms_df.loc[i]["w_bc_sp_opt"] = float(np.array(w_bc_sp_min)[0])
    parms_df.loc[i]["w_bc_st_opt"] = float(np.array(w_bc_st_min)[0])
    parms_df.loc[i][others_col] = others_min.astype("float").values[0,:]
  # Save best parameters
  parms_df = parms_df.reset_index(drop = True)
  parms_df.to_csv('%s/best_parms_MackeyGlass.csv' % (output_dir))
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #


# Load parameteres
parms_df = pd.read_csv('%s/best_parms_MackeyGlass.csv' % (output_dir))

# See the best results
tax_best_df = parms_df.iloc[parms_df["test_pred"].idxmax()]
# Do Macky-Glass task
cloop_best_single = cloop_simplex_reservoir(tax_best_df["Taxa"],
                       reservoir_ts, reservoir_db_name,
                       train_data, train_true, test_data, test_true,
                       w_in_sparsity = tax_best_df["w_in_sp_opt"],
                       w_in_strength = tax_best_df["w_in_st_opt"],
                       w_back_sparsity = tax_best_df["w_bc_sp_opt"],
                       w_back_strength = tax_best_df["w_bc_st_opt"],
                       Win_seed = 1234, Wback_seed = 1235, return_obj="class")
#tax_best_df
cloop_best_single.result_summary_df
cloop_best_single.result_summary_df.to_csv("%s/ERCMackeyGlass_SingleBest.csv" % output_dir)
# Save output
joblib.dump(cloop_best_single, "%s/ERCMackeyGlass_SingleBest.jb" % (output_dir), compress=3)





# Perform multinetwork approach ------------------------------------------------------------------- #
reservoir_var, reservoir_ts, reservoir_db_name = "Prok_Taxa00004", ecol_ts, "ecol_ts"

for n_spp in [500]:
  step = 30
  # Select variables
  ecol_ts_colnames = ecol_ts.iloc[:,1:].columns
  reservoir_var_list = pd.Index(parms_df["Taxa"])
  sp_opt_var = pd.Index(parms_df["w_in_sp_opt"])
  st_opt_var = pd.Index(parms_df["w_in_st_opt"])
  bc_sp_opt_var = pd.Index(parms_df["w_bc_sp_opt"])
  bc_st_opt_var = pd.Index(parms_df["w_bc_st_opt"])
  # Multiprocess computing
  par_erc = joblib.Parallel(n_jobs=-1, verbose = 5)([joblib.delayed(cloop_simplex_reservoir)
                (tax_i, reservoir_ts, reservoir_db_name,
                 train_data, train_true, test_data, test_true,
                 w_in_sparsity = x, w_in_strength = y,
                 w_back_sparsity = z, w_back_strength = l, return_obj="class")
                 for tax_i, x, y, z, l in zip(reservoir_var_list, sp_opt_var, st_opt_var, bc_sp_opt_var, bc_st_opt_var)])
  output_erc = par_erc[0].result_summary_df
  for i in range(1,len(par_erc)): output_erc = output_erc.append(par_erc[i].result_summary_df)
  # Check maximum prediction skill when single species reservoir is used --------------------- #
  output_erc = output_erc.reset_index(drop = True)
  
  # Prepare output dataframe
  multinetwork_df = pd.DataFrame()
  for n_network in np.append(np.arange(0, output_erc.shape[0], step), n_spp-1):
    # Combine reservoir states ----------------------------------------------------------------- #
    combined_reservoir_state = []
    total_E = 0
    if n_network == 0:
          combined_reservoir_state = par_erc[0].record_reservoir_nodes
          total_E = total_E + par_erc[0].num_nodes
    else:
      for sp_i in range(n_network + 1):
        if len(combined_reservoir_state) == 0:
          combined_reservoir_state = par_erc[sp_i].record_reservoir_nodes
          total_E = total_E + par_erc[sp_i].num_nodes
        else:
          combined_reservoir_state = np.hstack([combined_reservoir_state, par_erc[sp_i].record_reservoir_nodes])
          total_E = total_E + par_erc[sp_i].num_nodes
    # ------------------------------------------------------------------------------------------ #
    
    # Perform multiNetwork analysis
    msr01 = msr.MultinetworkSimplexReservoir("Combined_Network")
    msr01.learn_model(combined_reservoir_state, par_erc[0].train_true, ridge_lambda = 1)
    if n_network == 0:
      msr01.predict(par_erc[0].test_true, par_erc[0])
    else:
      msr01.predict(par_erc[0].test_true, par_erc[0:(n_network+1)])
    msr01.summarize_stat()
    msr01.result_summary_df["n_network"] = n_network + 1
    msr01.result_summary_df["E"] = total_E
    msr01.result_summary_df = pd.concat([msr01.result_summary_df, par_erc[0].result_summary_df.iloc[:,10:15]], axis = 1)
    multinetwork_df = multinetwork_df.append(msr01.result_summary_df)
  
  multinetwork_df = multinetwork_df.reset_index(drop = True)
  multinetwork_df.to_csv("%s/MultiERC_MackeyGlass.csv" % (output_dir))
multinetwork_df.iloc[:,0:5]
# ------------------------------------------------------------------------------------------ #
