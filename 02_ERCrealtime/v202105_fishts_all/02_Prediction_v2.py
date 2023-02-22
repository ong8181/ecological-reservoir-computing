####
#### Real-time Ecological Reservoir Computing
#### Predict near future
####

# Import essential modules
import numpy as np
import pandas as pd
import itertools; import joblib; import time
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import custom module
import module.rt_ecological_reservoir_ridge_v3 as rt

# Create output directory
import os; output_dir = "02_PredictionOut"; os.mkdir(output_dir)


# -------------------- Prepare data and set parameters -------------------- #
# Load data
ecol_ts = pd.read_csv('./01_LoadAllDataOut/d_all.csv')[0:] #(1319)
ecol_ts.columns
ecol_ts.shape

# Set parameters
train_fraction = 1/2
object_save = 95 # 17 weeks = 4 months
mat_set = 1
input_col = "temperature"
ridge_lambda = 1
pred_tp_range = np.arange(0, 120, 1)
fig_ylim = 2

# Compile state matrix
if mat_set == 1:
  all_state_matrix = np.array(ecol_ts)[:,[3,4,5,6,7,8]]
elif mat_set == 2:
  train_state_matrix = np.array(ecol_ts)[:,[3,5,7]]
  test_state_matrix = np.array(ecol_ts)[:,[4,6,8]]

# Prepare output dataframe
rtERC_output_all1 = pd.DataFrame()

# Main loop
for pred_tp in pred_tp_range:
  # Compile data
  if mat_set == 1:
    if pred_tp == 0:
      state_matrix = all_state_matrix.astype(float)
      lead_input = np.array(ecol_ts[input_col])
    else:
      state_matrix = all_state_matrix[:-pred_tp,].astype(float)
      lead_input = np.array(ecol_ts[input_col].iloc[:-pred_tp])
    # Compile data
    train_data_used = round(state_matrix.shape[0]*train_fraction)
    state_matrix1 = state_matrix[:train_data_used]
    state_matrix2 = state_matrix[train_data_used:]
    pred_input = np.array(ecol_ts[input_col].iloc[pred_tp:])
    pred_input1 = pred_input[:train_data_used]
    pred_input2 = pred_input[train_data_used:]
    lead_input1 = lead_input[:train_data_used]
    lead_input2 = lead_input[train_data_used:]
  else:
    if pred_tp == 0:
      lead_input1 = lead_input2 = np.array(ecol_ts[input_col])
    else:
      lead_input1 = lead_input2 = np.array(ecol_ts[input_col].iloc[:-pred_tp])
    state_matrix1 = train_state_matrix[pred_tp:,].astype(float)
    state_matrix2 = test_state_matrix[pred_tp:,].astype(float)
    pred_input = np.array(ecol_ts[input_col].iloc[pred_tp:])
    pred_input1 = np.array(ecol_ts[input_col].iloc[pred_tp:])
    pred_input2 = np.array(ecol_ts[input_col].iloc[pred_tp:])
  
  # Run rtERC
  rt1 = rt.rtERC("Tetrahymena Reservoir")
  rt1.learn_model(state_matrix1, pred_input1,
                  washout_fraction = 0, ridge_lambda = ridge_lambda)
  rt1.predict(state_matrix2, pred_input2)
  rt1.summarize_stat()
  
  # Perform simple linear regression (temporal autocorrelation)
  rt1.result_summary_df["E"] = state_matrix1.shape[1]
  rt1.result_summary_df["pred_tp"] = pred_tp
  rt1.result_summary_df["temporal_cor"] = np.corrcoef(lead_input2, pred_input2)[1,0]
  rt1.result_summary_df["temporal_r2"] = rt1.result_summary_df["temporal_cor"] ** 2
  
  # Perform a ridge regression
  temp_ridge = Ridge(alpha = ridge_lambda)
  temp_ridge.fit(lead_input1.reshape(-1,1), pred_input1.reshape(-1,1))
  ## For training data
  temp_ridge.train_predicted = temp_ridge.predict(lead_input1.reshape(-1,1))
  temp_ridge.train_ridge_score = temp_ridge.score(lead_input1.reshape(-1,1), pred_input1.reshape(-1,1)) # R^2
  temp_ridge.train_cor = np.corrcoef(temp_ridge.train_predicted.flatten(), pred_input1)[1,0]
  temp_ridge.train_r2 = temp_ridge.train_cor**2
  temp_ridge.train_rmse = np.sqrt(np.mean((np.array(pred_input1.reshape(-1,1) - temp_ridge.train_predicted))**2))
  temp_ridge.train_nmse = sum((pred_input1.reshape(-1,1) - temp_ridge.train_predicted)**2)/sum(pred_input1.reshape(-1,1)**2)
  ## For testing data
  temp_ridge.test_predicted = temp_ridge.predict(lead_input2.reshape(-1,1))
  temp_ridge.test_ridge_score = temp_ridge.score(lead_input2.reshape(-1,1), pred_input2.reshape(-1,1)) # R^2
  temp_ridge.test_cor = np.corrcoef(temp_ridge.test_predicted.flatten(), pred_input2)[1,0]
  temp_ridge.test_r2 = temp_ridge.test_cor**2
  temp_ridge.test_rmse = np.sqrt(np.mean((np.array(pred_input2.reshape(-1,1) - temp_ridge.test_predicted))**2))
  temp_ridge.test_nmse = sum((pred_input2.reshape(-1,1) - temp_ridge.test_predicted)**2)/sum(pred_input2.reshape(-1,1)**2)
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
  if pred_tp == object_save:
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
erc_all1 <- read.csv(sprintf("%s/AutomatedERC_output_all.csv", py$output_dir))
erc_all1$pred_cor <- erc_all1$test_cor
erc_all1$pred_cor[erc_all1$pred_cor < 0] <- 0
erc_all1$pred_r2 <- erc_all1$pred_cor^2

# ggplot
erc_long1 = pivot_longer(erc_all1[,c("pred_tp", "reservoir_E", "test_cor", "train_cor", "pred_cor", "pred_r2", "test_rmse", "test_nmse",
                                     "temporal_r2", "ridge_test_r2", "ridge_test_rmse", "ridge_test_nmse")],
                        cols = -c(pred_tp, reservoir_E), names_to = "train_or_test", values_to = "accuracy")
erc_long2 = erc_long1 %>% filter(train_or_test == "test_nmse" | train_or_test  == "ridge_test_nmse")

g1 = ggplot(erc_long2, aes(x = pred_tp, y = accuracy, color = train_or_test)) +
             geom_point(size = 1, alpha = 0.8) +
             geom_line(aes(linetype = train_or_test), size = 0.5, alpha = 0.8) +
             geom_hline(yintercept = 0, linetype = 2) +
             xlab("Prediction (min)") + ylab("NMSE") + geom_hline(yintercept = 1, linetype = 2) +
             scale_color_manual(values = c("royalblue", "red3"), labels = c("Linear readout", "Reservoir")) +
             scale_linetype(labels = c("Linear readout", "Reservoir")) +
             labs(color = NULL, linetype = NULL) +
             ggtitle("Tetrahymena reservoir") +
             theme(legend.position = c(0.5,0.2)) +
             ylim(0, py$fig_ylim) +
             NULL

ggsave(paste0(py$output_dir, "/Tetrahymena_MC.pdf"), plot = g1, width = 8, height = 4)
#head(erc_all1, 40)
g1

# Show prediction-observed plot
test_pred_df <- data.frame(time = 1:length(py$rt_save$test_true),
                           obs = py$rt_save$test_true,
                           pred = py$rt_save$test_predicted)
test_pred_df$lead_obs <- dplyr::lead(test_pred_df$obs, n = py$object_save)
g2 <- ggplot(test_pred_df, aes(x = obs, y = pred)) +
       geom_point() + geom_abline(slope = 1, intercept = 0, linetype = 2, color = "red3") +
       xlab("Observed") + ylab("Predicted") +
       NULL
g3 <- ggplot(test_pred_df, aes(x = obs, y = lead_obs))  +
       geom_point() + geom_abline(slope = 1, intercept = 0, linetype = 2, color = "red3") +
       xlab("Observed") + ylab("Observed (lead)") +
       NULL
test_pred_long <- test_pred_df[,c("time", "obs", "pred")] %>% pivot_longer(cols = -time)
g4 <- ggplot(test_pred_long, aes(x = time, y = value, color = name)) +
       geom_point(alpha = 0.8) + geom_line(alpha = 0.5) + 
       scale_color_manual(values = c("gray40", "red3"),
                          labels = c("Observed", "Predicted"),
                          name = NULL) +
       ggtitle(sprintf("Prediction = %s min future", py$object_save))
g_all1 <- plot_grid(g2, g3, ncol = 2)
g_all2 <- plot_grid(g_all1, g4, ncol = 1)

ggsave(paste0(py$output_dir, "/Tetrahymena_PredictionPlot.pdf"),
plot = g_all2, width = 8, height = 8)


# -------------------------------------------------- #
#                 Save objects
# -------------------------------------------------- #
fig_all <- list(g1, g2, g3, g4)
parm_all <- list(py$train_fraction, py$object_save, py$mat_set, py$ridge_lambda, py$pred_tp_range)
names(parm_all) <- c("train_fraction", "object_save", "mat_set", "ridge_lambda", "pred_tp_range")

saveRDS(fig_all, sprintf("%s/FigObject.obj", py$output_dir))
saveRDS(parm_all, sprintf("%s/ParameterList.obj", py$output_dir))
saveRDS(r_df, sprintf("%s/ReservoirState.obj", py$output_dir))
saveRDS(erc_all1, sprintf("%s/ReservoirResult.obj", py$output_dir))
saveRDS(py$rt_save, sprintf("%s/ResultExample.obj", py$output_dir))

