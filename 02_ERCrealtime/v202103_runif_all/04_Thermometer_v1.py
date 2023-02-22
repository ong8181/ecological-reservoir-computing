####
#### Real-time Ecological Reservoir Computing
#### No.4 Thermometer analysis
####

# Import essential modules
import numpy as np
import pandas as pd
import itertools; import joblib; import time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Set pandas options
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)

# Import custom module
import module.rt_ecological_reservoir_ridge_v3 as rt

# Create output directory
import os; output_dir = "04_ThermometerOut"; os.mkdir(output_dir)

# -------------------- Prepare data and set parameters -------------------- #
# Load data
ecol_ts0 = pd.read_csv('./01_LoadAllDataOut/d_all.csv')[0:] #(1319)
ecol_ts = pd.read_csv('./01_LoadAllDataOut/d_all.csv')[819:] #(1319)
ecol_ts.columns
ecol_ts.shape

train_fraction = 1/2
object_save = 5
mat_set = 2
input_col = "temperature"
# Set additional parameters
ridge_lambda = 1
delay_tp_range = np.arange(0, 60, 1)
fig_ylim = 0.7

# Run column number
#run1 = 3:7
#run2 = 8:12
#run3 = 13:17
#run4 = 18:22
#run5 = 23:27
#run6 = 28:32
#run7 = 33:37
# Compile state matrix
if mat_set == 1:
  all_state_matrix = np.array(ecol_ts)[:,range(2,3)]
elif mat_set == 2:
  train_state_matrix = np.array(ecol_ts)[:,[3,5,7]]
  test_state_matrix = np.array(ecol_ts)[:,[4,6,8]]

# Prepare output dataframe
rtERC_output_all1 = pd.DataFrame()


# Main loop
for delay_tp in delay_tp_range:
  # Compile data
  if mat_set == 1:
    if delay_tp == 0:
      state_matrix = all_state_matrix.astype(float)
      unlagged_input = np.array(ecol_ts[input_col])
      lagged_input = np.array(ecol_ts[input_col])
    else:
      state_matrix = all_state_matrix[delay_tp:,].astype(float)
      unlagged_input = np.array(ecol_ts[input_col].iloc[delay_tp:])
      lagged_input = np.array(ecol_ts[input_col].iloc[:-delay_tp])
    # Compile data
    train_data_used = round(state_matrix.shape[0]*train_fraction)
    state_matrix1 = state_matrix[:train_data_used]
    state_matrix2 = state_matrix[train_data_used:]
    lagged_input1 = lagged_input[:train_data_used]
    lagged_input2 = lagged_input[train_data_used:]
    unlagged_input1 = unlagged_input[:train_data_used]
    unlagged_input2 = unlagged_input[train_data_used:]
  else:
    if delay_tp == 0:
      state_matrix1 = train_state_matrix
      state_matrix2 = test_state_matrix
      unlagged_input1 = unlagged_input2 = np.array(ecol_ts[input_col])
      lagged_input = lagged_input1 = lagged_input2 = np.array(ecol_ts[input_col])
    else:
      state_matrix1 = train_state_matrix[delay_tp:,].astype(float)
      state_matrix2 = test_state_matrix[delay_tp:,].astype(float)
      unlagged_input1 = unlagged_input2 = np.array(ecol_ts[input_col].iloc[delay_tp:])
      lagged_input = lagged_input1 = lagged_input2 = np.array(ecol_ts[input_col].iloc[:-delay_tp])
  
  # Run rtERC
  rt1 = rt.rtERC("Tetrahymena Reservoir")
  # Use half as traininig data and the remaining halsf as testing data
  rt1.learn_model(state_matrix1, lagged_input1,
                  washout_fraction = 0, ridge_lambda = ridge_lambda)
  rt1.predict(state_matrix2, lagged_input2)
  rt1.summarize_stat()
  
  # Perform temporal autocorrelation
  rt1.result_summary_df["E"] = state_matrix1.shape[1]
  rt1.result_summary_df["delay_tp"] = delay_tp
  rt1.result_summary_df["temporal_cor"] = np.corrcoef(unlagged_input2, lagged_input2)[1,0]
  rt1.result_summary_df["temporal_r2"] = rt1.result_summary_df["temporal_cor"] ** 2
  rt1.result_summary_df["temporal_rmse"] = np.sqrt(np.mean((np.array(unlagged_input2 - lagged_input2))**2))
  rt1.result_summary_df["temporal_nmse"] = sum((unlagged_input2 - lagged_input2)**2)/sum(unlagged_input2**2)
  
  # Perform a ridge regression
  temp_ridge = Ridge(alpha = ridge_lambda)
  temp_ridge.fit(unlagged_input1.reshape(-1,1), lagged_input1.reshape(-1,1))
  ## For training data
  temp_ridge.train_predicted = temp_ridge.predict(unlagged_input1.reshape(-1,1))
  temp_ridge.train_ridge_score = temp_ridge.score(unlagged_input1.reshape(-1,1), lagged_input1.reshape(-1,1)) # R^2
  temp_ridge.train_cor = np.corrcoef(temp_ridge.train_predicted.flatten(), lagged_input1)[1,0]
  temp_ridge.train_r2 = temp_ridge.train_cor**2
  temp_ridge.train_rmse = np.sqrt(np.mean((np.array(lagged_input1.reshape(-1,1) - temp_ridge.train_predicted))**2))
  temp_ridge.train_nmse = sum((lagged_input1.reshape(-1,1) - temp_ridge.train_predicted)**2)/sum(lagged_input1.reshape(-1,1)**2)
  ## For testing data
  temp_ridge.test_predicted = temp_ridge.predict(unlagged_input2.reshape(-1,1))
  temp_ridge.test_ridge_score = temp_ridge.score(unlagged_input2.reshape(-1,1), lagged_input2.reshape(-1,1)) # R^2
  temp_ridge.test_cor = np.corrcoef(temp_ridge.test_predicted.flatten(), lagged_input2)[1,0]
  temp_ridge.test_r2 = temp_ridge.test_cor**2
  temp_ridge.test_rmse = np.sqrt(np.mean((np.array(lagged_input2.reshape(-1,1) - temp_ridge.test_predicted))**2))
  temp_ridge.test_nmse = sum((lagged_input2.reshape(-1,1) - temp_ridge.test_predicted)**2)/sum(lagged_input2.reshape(-1,1)**2)
  ## Add statistics to the summary table
  ### Traning data
  rt1.result_summary_df["ridge_train_cor"] = temp_ridge.train_cor
  rt1.result_summary_df["ridge_train_r2"] = temp_ridge.train_r2
  rt1.result_summary_df["ridge_train_score"] = temp_ridge.train_cor
  rt1.result_summary_df["ridge_train_rmse"] = temp_ridge.train_rmse
  rt1.result_summary_df["ridge_train_nmse"] = temp_ridge.train_nmse
  ### Test data
  rt1.result_summary_df["ridge_test_cor"] = temp_ridge.test_cor
  rt1.result_summary_df["ridge_test_r2"] = temp_ridge.test_r2
  rt1.result_summary_df["ridge_test_rmse"] = temp_ridge.test_rmse
  rt1.result_summary_df["ridge_test_nmse"] = temp_ridge.test_nmse
  
  # Save and append resultss
  if delay_tp == object_save:
    rt_save = rt1
  rtERC_output_all1 = rtERC_output_all1.append(rt1.result_summary_df)

rtERC_output_all1 = rtERC_output_all1.reset_index(drop = True)
rtERC_output_all1.to_csv("%s/AutomatedERC_output_all.csv" % (output_dir), index = False)

exit


# -------------------------------------------------- #
#                  R visualization
# -------------------------------------------------- #
library(reticulate); packageVersion("reticulate") # 1.20, 2021.8.20
library(tidyverse); packageVersion("tidyverse") # 1.3.1, 2021.8.20
library(cowplot); packageVersion("cowplot") # 1.1.1, 2021.8.20
library(ggsci); packageVersion("ggsci") # 2.9, 2021.8.20
theme_set(theme_cowplot())

# Load data
r_df <- py$ecol_ts
r_df0 <- py$ecol_ts0
erc_all1 <- read.csv(sprintf("%s/AutomatedERC_output_all.csv", py$output_dir))
erc_all1$test_cor[erc_all1$test_cor < 0] <- 0
erc_all1$memory_capacity <- erc_all1$test_cor^2

# ggplot
erc_long1 = pivot_longer(erc_all1[,c("delay_tp", "reservoir_E", "test_cor", "train_cor", "memory_capacity",
                                     "temporal_r2", "temporal_nmse", "test_nmse",
                                     "ridge_test_nmse", "ridge_test_r2")],
                        cols = -c(delay_tp, reservoir_E), names_to = "train_or_test", values_to = "accuracy")
erc_long2 = erc_long1 %>% filter(train_or_test == "memory_capacity" | train_or_test == "temporal_r2" | train_or_test == "ridge_test_r2")
erc_long3 = erc_long1 %>% filter(train_or_test == "ridge_test_r2" | train_or_test == "test_nmse")

g1 = ggplot(erc_long2, aes(x = delay_tp, y = accuracy, color = train_or_test)) +
             geom_point(size = 1, alpha = 0.8) +
             geom_line(aes(linetype = train_or_test), size = 0.5, alpha = 0.8) +
             geom_hline(yintercept = 0, linetype = 2) +
             xlab("Delay (min)") + ylab(expression(paste("Coef. of determination (", R^{2}, ")"))) +
             scale_color_manual(values = c("red3", "royalblue", "gray"), labels = c("Reservoir", "Ridge regression", "Temporal correlation")) +
             scale_linetype(labels = c("Reservoir", "Ridge regression", "Temporal correlation")) +
             labs(color = NULL, linetype = NULL) +
             ggtitle("Tetrahymena reservoir") +
             theme(legend.position = c(0.5,0.8)) +
             ylim(0, py$fig_ylim) +
             NULL

g1_2 = ggplot(erc_long3, aes(x = delay_tp, y = accuracy, color = train_or_test)) +
             geom_point(size = 1, alpha = 0.8) +
             geom_line(aes(linetype = train_or_test), size = 0.5, alpha = 0.8) +
             geom_hline(yintercept = 1, linetype = 2) +
             xlab("Delay (min)") + ylab("Error") +
             scale_color_manual(values = c("red3", "royalblue"), labels = c("ridge_test_nmse", "test_nmse")) +
             scale_linetype(labels = c("ridge_test_nmse", "test_nmse")) +
             labs(color = NULL, linetype = NULL) +
             ggtitle("Tetrahymena reservoir") +
             theme(legend.position = c(0.5,0.2)) +
             ylim(0, 2.5) +
             NULL

ggsave(paste0(py$output_dir, "/Tetrahymena_MC.pdf"), plot = g1, width = 6, height = 4)
head(erc_all1, 40)
g1

# Show prediction-observed plot
test_pred_df <- data.frame(time = 1:length(py$rt_save$test_true),
                           obs = py$rt_save$test_true,
                           pred = py$rt_save$test_predicted)
test_pred_df$lagged_obs <- dplyr::lag(test_pred_df$obs, n = py$object_save)
g2 <- ggplot(test_pred_df, aes(x = obs, y = pred)) +
       geom_point() + geom_abline(slope = 1, intercept = 0, linetype = 2, color = "red3") +
       xlab("Observed") + ylab("Predicted") +
       NULL
g3 <- ggplot(test_pred_df, aes(x = obs, y = lagged_obs))  +
       geom_point() + geom_abline(slope = 1, intercept = 0, linetype = 2, color = "red3") +
       xlab("Observed") + ylab("Observed (lagged)") +
       NULL
test_pred_long <- test_pred_df[,c("time", "obs", "pred")] %>% pivot_longer(cols = -time)
g4 <- ggplot(test_pred_long, aes(x = time, y = value, color = name)) +
       geom_point(alpha = 0.8) + geom_line(alpha = 0.5) + 
       scale_color_manual(values = c("gray40", "red3"),
                          labels = c("Observed", "Predicted"),
                          name = NULL) +
       ggtitle(sprintf("Delay = %s", py$object_save))
g_all1 <- plot_grid(g2, g3, ncol = 2)
g_all2 <- plot_grid(g_all1, g4, ncol = 1)

ggsave(paste0(py$output_dir, "/Tetrahymena_PredictionPlot.pdf"),
plot = g_all2, width = 8, height = 8)


# -------------------------------------------------- #
#                 Save objects
# -------------------------------------------------- #
fig_all <- list(g1, g2, g3, g4)
parm_all <- list(py$train_fraction, py$object_save, py$mat_set, py$ridge_lambda, py$delay_tp_range)
names(parm_all) <- c("train_fraction", "object_save", "mat_set", "ridge_lambda", "delay_tp_range")

saveRDS(fig_all, sprintf("%s/FigObject.obj", py$output_dir))
saveRDS(parm_all, sprintf("%s/ParameterList.obj", py$output_dir))
saveRDS(r_df, sprintf("%s/ReservoirState.obj", py$output_dir))
saveRDS(r_df0, sprintf("%s/ReservoirState0.obj", py$output_dir))
saveRDS(erc_all1, sprintf("%s/ReservoirResult.obj", py$output_dir))
saveRDS(py$rt_save, sprintf("%s/ResultExample.obj", py$output_dir))
