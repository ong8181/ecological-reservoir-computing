####
#### Reservoir computing
#### No.11: Calculating ESP index for simplex reservoir
####

# Import essential modules
import numpy as np
import pandas as pd
import time; import joblib
from scipy import linalg

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import custom module
import module.predict_by_ecological_reservoir_v1 as erc

# Create output directory
import os; output_dir = "11_ESPindex_TSpredOut"; os.mkdir(output_dir)

# -------------------- 0. Set global parameters -------------------- #
# Prepare output dataframe
#output_all_df = pd.DataFrame()

# Load time series
lorenz = pd.read_csv('./data/LorenzTS_tau10.csv')
ecol_ts = pd.read_csv('./data/edna_asv_table_prok_all.csv')
#fish_ts = pd.read_csv('./data/BCreation_s0001_weekly.csv')
rand_ts = pd.read_csv('./data/runif_ts.csv', index_col = 0)
rand_ts2 = pd.read_csv('./data/runif_ts2.csv', index_col = 0)
best_e = pd.read_csv("./data/bestE_all.csv")

# Select target data
target_ts, target_var, target_db_name, test_fraction = rand_ts2[0:3000], 'random_01', 'random', 0.2

# Load the best results
## Select the top N species
n_sp1 = 500
n_sp2 = 47
rep_per_sp = 5
## Top 100 prokaryote names
prok_dom = ecol_ts.sum(axis=0)[1:][np.argsort(ecol_ts.sum(axis=0)[1:])[::-1]][:n_sp1]
prok_var = prok_dom.index
fish_dom = fish_ts.sum(axis=0)[1:][np.argsort(fish_ts.sum(axis=0)[1:])[::-1]][:n_sp2]
fish_var = fish_dom.index

# ------------------------------ Define function ------------------------------ #
def calculate_ESP(reservoir_var, reservoir_ts, reservoir_db_name,
                  n_nodes = None, n_nn = None, w_in_strength = 2, w_in_sparsity = 0,
                  bestE_data = best_e, rand_seed = 0):
  if n_nodes == None:
    E = int(bestE_data[bestE_data["variable"] == reservoir_var]["bestE"])
  else:
    E = n_nodes
  # Calculate the first reservoir states
  erc1 = erc.SimplexReservoir(reservoir_var, reservoir_ts, reservoir_db_name)
  erc1.compile_reservoir_data(E)
  erc1.prepare_target_data(target_ts, target_var, target_db_name, test_fraction)
  erc1.initialize_reservoir(w_in_strength = w_in_strength, w_in_sparsity = w_in_sparsity, n_nn = n_nn)
  
  # Set random initial indices
  rand_id_cand = []
  for s in np.where(np.sum(np.isnan(erc1.db), axis = 1) == 0): rand_id_cand.extend(s)
  np.random.seed(rand_seed); random_index = np.random.choice(rand_id_cand, 2, replace = False)
  
  # Calculate distances of reservoir states
  erc1.compute_reservoir_state(initial_method = 'manual', manual_index = random_index[0])
  
  # Calculate the second reservoir states
  erc2 = erc.SimplexReservoir(reservoir_var, reservoir_ts, reservoir_db_name)
  erc2.compile_reservoir_data(E)
  erc2.prepare_target_data(target_ts, target_var, target_db_name, test_fraction)
  erc2.initialize_reservoir(w_in_strength = w_in_strength, w_in_sparsity = w_in_sparsity, n_nn = n_nn)
  erc2.compute_reservoir_state(initial_method = 'manual', manual_index = random_index[1])
  
  # Calculate distance between the two reservoir states
  node_dist = np.sqrt(np.power(erc1.record_reservoir_nodes - erc2.record_reservoir_nodes, 2).sum(axis = 1))
  node_dist_df = pd.DataFrame(node_dist.T)
  
  return node_dist_df
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Calculate ESP for a specific species
# --------------------------------------------------------------------------- #
node_dist_prok = pd.DataFrame() # For prokayote reservoir
node_dist_fish = pd.DataFrame() # For fish reservoir

for rand_seed in range(100):
  node_dist_prok_tmp = calculate_ESP(prok_var[0], ecol_ts, 'ecol_ts', rand_seed = rand_seed)
  node_dist_fish_tmp = calculate_ESP(fish_var[0], fish_ts, 'fish_ts', rand_seed = rand_seed)
  # Compile data for visualization
  node_dist_prok = pd.concat([node_dist_prok, node_dist_prok_tmp], axis = 1)
  node_dist_fish = pd.concat([node_dist_fish, node_dist_fish_tmp], axis = 1)

# Prepare output data
node_dist_prok.columns = pd.Index(['thread_%03d' % (i) for i in range(1, int(node_dist_prok.shape[1]) + 1)])
node_dist_fish.columns = pd.Index(['thread_%03d' % (i) for i in range(1, int(node_dist_fish.shape[1]) + 1)])
node_dist_prok['state_id'] = np.arange(1, (node_dist_prok.shape[0] + 1))
node_dist_fish['state_id'] = np.arange(1, (node_dist_fish.shape[0] + 1))

# Save result to pickel
node_dist_prok.to_csv("%s/ESPindex_Prok_Random_1sp_rep100.csv" % output_dir)
node_dist_fish.to_csv("%s/ESPindex_Fish_Random_1sp_rep100.csv" % output_dir)


# --------------------------------------------------------------------------- #
# Calculate ESP for all prokaryote and fish time series
# --------------------------------------------------------------------------- #
# Prokaryote time series
## Prepare output data.frame for prokaryote
node_dist_prok_all = pd.DataFrame() # For prokayote reservoir

for prok_var_i in prok_var:
  node_dist_prok_all_tmp = pd.DataFrame() # For prokayote reservoir
  #prok_var_i = prok_var[0]
  for rand_seed in range(rep_per_sp):
    # Calculate state distance
    node_dist_prok_i_tmp = calculate_ESP(prok_var_i, ecol_ts, 'ecol_ts', rand_seed = rand_seed)
    node_dist_prok_all_tmp = pd.concat([node_dist_prok_all_tmp, node_dist_prok_i_tmp], axis = 1)
  # Prepare output data
  node_dist_prok_all_tmp.columns = pd.Index(['%s_thread_%03d' % (prok_var_i, i) for i in range(1, rep_per_sp+1)])
  node_dist_prok_all = pd.concat([node_dist_prok_all, node_dist_prok_all_tmp], axis = 1)

node_dist_prok_all['state_id'] = np.arange(1, (node_dist_prok_all.shape[0] + 1))

# Save result to pickel
node_dist_prok_all.to_csv("%s/ESPindex_Prok_RandomTS_top%d_rep%d.csv" % (output_dir, n_sp1, rep_per_sp))


# Fish time series
## Prepare output data.frame for fish
node_dist_fish_all = pd.DataFrame() # For fish reservoir

for fish_var_i in fish_var:
  node_dist_fish_all_tmp = pd.DataFrame() # For fish reservoir
  for rand_seed in range(rep_per_sp):
    # Calculate state distance
    node_dist_fish_i_tmp = calculate_ESP(fish_var_i, fish_ts, 'fish_ts', rand_seed = rand_seed)
    node_dist_fish_all_tmp = pd.concat([node_dist_fish_all_tmp, node_dist_fish_i_tmp], axis = 1)
  # Prepare output data
  node_dist_fish_all_tmp.columns = pd.Index(['%s_thread_%03d' % (fish_var_i, i) for i in range(1, rep_per_sp+1)])
  node_dist_fish_all = pd.concat([node_dist_fish_all, node_dist_fish_all_tmp], axis = 1)

node_dist_fish_all['state_id'] = np.arange(1, (node_dist_fish_all.shape[0] + 1))

# Save result to pickel
node_dist_fish_all.to_csv("%s/ESPindex_Fish_RandomTS_top%d_rep%d.csv" % (output_dir, n_sp2, rep_per_sp))



#============================= R Session ==============================#
# Visualize result using ggplot2 in R
exit

# Load library
library(reticulate)
library(tidyverse)
library(cowplot); theme_set(theme_cowplot())

# Save variable names
np <- import("numpy", convert = FALSE)
prok_var <- py_to_r(np$array(py$prok_var))
fish_var <- py_to_r(np$array(py$fish_var))
saveRDS(prok_var, sprintf("%s/prok_var_top%s.obj", py$output_dir, py$n_sp1))
saveRDS(fish_var, sprintf("%s/fish_var_top%s.obj", py$output_dir, py$n_sp2))


node_dist_melt <- pivot_longer(py$node_dist_prok, col = - state_id, names_to = "variable", values_to = "value")

g1 <- ggplot(node_dist_melt, aes(x = state_id, y = value, color = variable)) +
      geom_point(size = 0.3, alpha = 0.3) + geom_line(size = 0.3, alpha = 0.3) + theme(legend.position = "none") +
      xlab("State ID") + 
      ylab("Euclidean distance between two reservoir states") + scale_colour_viridis_d()
#========================================================================#
