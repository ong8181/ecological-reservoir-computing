####
#### Reservoir computing
#### No.2 Demonstration using logistic map (equation)
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
import module.predict_by_logistic_reservoir_v2 as lrt

# Create output directory
import os; output_dir = "02_LogisticRC_TSpredOut"; os.mkdir(output_dir)


# ----------------------------------------------- #
# Prepare data and set parameters
# ----------------------------------------------- #
# Load time series
lorenz = pd.read_csv('./data/LorenzTS_tau10.csv')

# Prepare output dataframe
output_all_df = pd.DataFrame()

# Set target variable
target_var, target_ts = 'x', lorenz[91:]

# Netrowk parameters
num_reservoir_nodes = 2 # Equals to the number of variables in the logistic map
#num_input_nodes = 1 # Input is a scalar value
#num_output_nodes = 1 # Output is a scalar vlue

# Specify the test fraction
test_fraction = 0.2

# Parameter search
rx_range = np.arange(2.5, 3.31, 0.1)
ry_range = np.arange(2.5, 3.31, 0.1)
bxy_range = np.arange(-0.2, 0.21, 0.05)
byx_range = np.arange(-0.2, 0.21, 0.05)


# ----------------------------------------------- #
# Search rx and ry
# ----------------------------------------------- #
output_all_df_rx_ry = pd.DataFrame()
for rx, ry in itertools.product(rx_range, ry_range):
  lrc = lrt.LogisticReservoir(network_name = "Logistic Map")
  lrc.prepare_target_data(target_ts, target_var, test_fraction)
  lrc.initialize_reservoir(num_reservoir_nodes, w_in_sparsity = 0.1, w_in_strength = 0.2)
  lrc.compute_reservoir_state(rx = rx, ry = ry, bxy = 0.1, byx = -0.2)
  lrc.learn_model(ridge_lambda = 0.05)
  lrc.predict()
  lrc.summarize_stat()
  
  # Append summary dataframe
  output_all_df_rx_ry = output_all_df_rx_ry.append(lrc.result_summary_df)
output_all_df_rx_ry = output_all_df_rx_ry.reset_index(drop = True)
# --------------------------------------------------------------------------- #
output_all_df_rx_ry.to_csv("02_LogisticRC_TSpredOut/RxRy_test2.csv")


# ----------------------------------------------- #
# Search bxy and byx
# ----------------------------------------------- #
output_all_df_bxy_byx = pd.DataFrame()
for bxy, byx in itertools.product(bxy_range, byx_range):
  lrc = lrt.LogisticReservoir(network_name = "Logistic Map")
  lrc.prepare_target_data(target_ts, target_var, test_fraction)
  lrc.initialize_reservoir(num_reservoir_nodes, w_in_sparsity = 0.1, w_in_strength = 0.2)
  lrc.compute_reservoir_state(rx = 2.9, ry = 2.92, bxy = bxy, byx = byx)
  lrc.learn_model(ridge_lambda = 0.05)
  lrc.predict()
  lrc.summarize_stat()
  
  # Append summary dataframe
  output_all_df_bxy_byx = output_all_df_bxy_byx.append(lrc.result_summary_df)
output_all_df_bxy_byx = output_all_df_bxy_byx.reset_index(drop = True)
# --------------------------------------------------------------------------- #
output_all_df_bxy_byx.to_csv("%s/BxyByx_test.csv" % output_dir)


# ----------------------------------------------- #
# Search all
# ----------------------------------------------- #
output_all_df_all = pd.DataFrame()
for rx, ry, bxy, byx in itertools.product(rx_range, ry_range, bxy_range, byx_range):
  lrc = lrt.LogisticReservoir(network_name = "Logistic Map")
  lrc.prepare_target_data(target_ts, target_var, test_fraction)
  lrc.initialize_reservoir(num_reservoir_nodes, w_in_sparsity = 0.1, w_in_strength = 0.2)
  lrc.compute_reservoir_state(rx = rx, ry = ry, bxy = bxy, byx = byx)
  lrc.learn_model(ridge_lambda = 0.05)
  lrc.predict()
  lrc.summarize_stat()
  # Append summary dataframe
  output_all_df_all = output_all_df_all.append(lrc.result_summary_df)
output_all_df_all = output_all_df_all.reset_index(drop = True)
# --------------------------------------------------------------------------- #
output_all_df_all.to_csv("02_LogisticRC_TSpredOut/All_test2.csv")
# Extract parameters
all_best = output_all_df_all.loc[output_all_df_all['NMSE_test'].min() == output_all_df_all['NMSE_test']]
rx0 = float(all_best["rx"]) # 3.0
ry0 = float(all_best["ry"]) # 2.7
bxy0 = float(all_best["bxy"]) # -0.2
byx0 = float(all_best["byx"]) #  0.2


# ----------------------------------------------- #
# Search w_in_sparsity and w_in_strength
# ----------------------------------------------- #
# For parallel analysis
def reservoir_computing_parallel(w_in_sparsity = 0.1, w_in_strength = 0.1,
                                 rx = rx0, ry = ry0, bxy = bxy0, byx = byx0):
  # Step.1: Initialize class "SimplexReservoir"
  par_lrc = lrt.LogisticReservoir(network_name = "Logistic Map")
  # Step.2: Prepare target data
  par_lrc.prepare_target_data(target_ts, target_var, test_fraction)
  # Step.3: Initialize reservoir
  par_lrc.initialize_reservoir(num_reservoir_nodes = 2, w_in_sparsity = w_in_sparsity, w_in_strength = w_in_strength)
  # Step.4: Training reservoir
  par_lrc.compute_reservoir_state(rx = rx, ry = ry, bxy = bxy, byx = byx)
  # Step.5: Learn weights by ridge regression
  par_lrc.learn_model(ridge_lambda = 0.05, washout_fraction = 0.05)
  # Step.6: Predict test data
  par_lrc.predict()
  # Step.7: Summarize stats
  par_lrc.summarize_stat()
  return par_lrc.result_summary_df
# ------------------------------------------------------------------------------------------ #

# Parameter search
WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WinSt_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

par_rc1 = joblib.Parallel(n_jobs=-3, verbose = 10)([joblib.delayed(reservoir_computing_parallel)
                                    (w_in_sparsity = x, w_in_strength = y) for x, y in itertools.product(WinSp_range, WinSt_range)])
output_all_df1 = par_rc1[0]
for j in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[j])
output_all_df1 = output_all_df1.reset_index(drop = True)
wspst_min = output_all_df1.loc[output_all_df1['NMSE_test'].min() == output_all_df1['NMSE_test']]
w_in_sp_best = float(np.array(wspst_min["Win_sparsity"])[0])
w_in_st_best = float(np.array(wspst_min["Win_strength"])[0])

# Save best parameters
output_all_df1 = output_all_df1.reset_index(drop = True); output_all_df1.to_csv('%s/WinSpSt_test.csv' % output_dir)

# Perform the best results
lrc = lrt.LogisticReservoir(network_name = "Logistic Map")
lrc.prepare_target_data(target_ts, target_var, test_fraction)
lrc.initialize_reservoir(num_reservoir_nodes, w_in_sparsity = w_in_sp_best, w_in_strength = w_in_st_best)
lrc.compute_reservoir_state(rx = rx0, ry = ry0, bxy = bxy0, byx = byx0)
lrc.learn_model(ridge_lambda = 0.05)
lrc.predict()
lrc.summarize_stat()
lrc.result_summary_df

# Save result to pickel
joblib.dump(lrc, "%s/LR_PredLorenz.jb" % output_dir, compress=3)


# ----------------------------------------------- #
# Perform simple ridge regression under the same condition
# ----------------------------------------------- #
pred_wo_reservoir = lrt.LogisticReservoir(network_name = "wo_reservoir")
pred_wo_reservoir.prepare_target_data(target_ts, target_var, test_fraction)
pred_wo_reservoir.learn_model_wo_reservoir(ridge_lambda = 0.05)
pred_wo_reservoir.predict_wo_reservoir()
joblib.dump(pred_wo_reservoir, "%s/LR_PredLorenz_wo_reservoir.jb" % output_dir, compress=3)
