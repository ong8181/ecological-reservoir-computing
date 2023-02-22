####
#### Figures: Logistic reservoir
####

# Load library
library(reticulate); packageVersion("reticulate") # 1.26, 2022.11.17
library(tidyverse); packageVersion("tidyverse") # 1.3.2, 2022.11.17
library(cowplot); packageVersion("cowplot") # 1.1.1
library(rEDM); packageVersion("rEDM") # 0.7.5
library(ggsci); packageVersion("ggsci") # 2.9
library(scales); packageVersion("scales") # 1.2.1, 2022.11.17
options('tibble.print_max' = 1000)
theme_set(theme_cowplot())

# Create figure output directory
fig_output_dir <- "0_RawFigs"
obj_output_dir <- "0_FigObj"
dir.create(fig_output_dir)
dir.create(obj_output_dir)
# Identify parent director and change working directory
path_element <- str_split(rstudioapi::getSourceEditorContext()$path, pattern = "/")[[1]]
my_dir <- path_element %>% .[-length(.)] %>% paste(collapse = "/")
parent_dir <- path_element %>% .[-c(length(.)-1, length(.))] %>% paste(collapse = "/")
setwd(parent_dir); getwd()
repl_python()

#::::::::::::::::::::::::::::::::::::::::::::::::::#
#::::::::::::::::: Python script ::::::::::::::::::#
# Load results from python objects
import joblib; import pandas as pd

# Logistic equation
logis_eqn = joblib.load("02_LogisticRC_TSpredOut/LR_PredLorenz.jb")
logis_wo_reservoir = joblib.load("02_LogisticRC_TSpredOut/LR_PredLorenz_wo_reservoir.jb")

# Logistic time series
logis_erc = joblib.load("03_LogisticSimplexRC_LorenzPredOut/LSR_PredLorenz.jb")
exit
#::::::::::::::::::::::::::::::::::::::::::::::::::#
#::::::::::::::::::::::::::::::::::::::::::::::::::#
# Reset working directory
setwd(my_dir)

# Load logistic reservoir results
logis_BxyByx <- read.csv("../02_LogisticRC_TSpredOut/BxyByx_test.csv")
logis_RxRy <- read.csv("../02_LogisticRC_TSpredOut/RxRy_test2.csv")
logis_WinSpSt <- read.csv("../02_LogisticRC_TSpredOut/WinSpSt_test.csv")
logis_erc_df1 <- read.csv("../03_LogisticSimplexRC_LorenzPredOut/LSR_output_df1.csv")
logis_erc_df2 <- read.csv("../03_LogisticSimplexRC_LorenzPredOut/LSR_output_df2.csv")

# Visualize the results of logistic equation reservoir ---------------------------------- #
# Patterns in the predicted values
eqn_df_train <- data.frame(time = 1:length(py$logis_eqn$train_predicted),
                           obs = py$logis_eqn$train_true[(py$logis_eqn$washout+1):length(py$logis_eqn$train_true)],
                           pred_res = py$logis_eqn$train_predicted,
                           pred_lm = py$logis_wo_reservoir$train_predicted)
eqn_df_test <- data.frame(time = 1:length(py$logis_eqn$test_true),
                          obs = py$logis_eqn$test_true,
                          pred_res = py$logis_eqn$test_predicted,
                          pred_lm = py$logis_wo_reservoir$test_predicted)
eqn_df_melt <- pivot_longer(eqn_df_test, cols = -1, names_to = "variable", values_to = "value")

cor(eqn_df_test$obs, eqn_df_test$pred_res)^2 # R2 = 0.847
mean(abs(eqn_df_test$pred_res - eqn_df_test$obs)) # MAE = 0.0698
sqrt(mean((eqn_df_test$pred_res - eqn_df_test$obs)^2)) # RMSE = 0.0897
cor(eqn_df_test$obs, eqn_df_test$pred_lm)^2  # R2 = 0.722
mean(abs(eqn_df_test$pred_lm - eqn_df_test$obs)) # MAE = 0.1031
sqrt(mean((eqn_df_test$pred_lm - eqn_df_test$obs)^2)) # RMSE = 0.1235

log_eqn_g1_1 <- ggplot(eqn_df_test, aes(x = obs, y = pred_res)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("Observed") + ylab("Predicted") + ggtitle("Logistic reservoir\n(Equation)")
log_eqn_g1_2 <- ggplot(eqn_df_test, aes(x = obs, y = pred_lm)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("Observed") + ylab("Predicted") + ggtitle("Simple ridge regression")
log_eqn_g2 <- ggplot(eqn_df_melt, aes(x = time, y = value, color = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("Logistic reservoir v.s. ridge regression") +
  scale_color_manual(values = c("black", "royalblue", "red3"), name = NULL,
                     labels = c("Observed",
                                "Predicted by a ridge regression",
                                "Predicted by a logistic map reservoir")) +
  scale_alpha_manual(values = c(1, 0.8, 0.8), name = NULL,
                     labels = c("Observed",
                                "Predicted by a ridge regression",
                                "Predicted by a logistic map reservoir")) +
  theme(legend.position = "right") +
  #scale_linetype_manual(values = c(1, 1, 1)) +
  xlab("Time") + ylab("Value")


# Parameter search
log_eqn_g3 <- ggplot(logis_BxyByx, aes(x = bxy, y = byx, z = rho_test)) +
  geom_tile(aes(fill=rho_test)) + xlab("bxy") + ylab("byx") +
  scale_fill_gradient(high="red3", low="white", name = expression(rho), limits=c(0, 1), oob = squish)
log_eqn_g4 <- ggplot(logis_RxRy, aes(x = rx, y = ry, z = rho_test)) +
  geom_tile(aes(fill=rho_test)) + xlab("rx") + ylab("ry") +
  scale_fill_gradient(high="red3", low="white", name = expression(rho), limits=c(0, 1), oob = squish)
log_eqn_g5 <- ggplot(logis_WinSpSt, aes(x = Win_sparsity, y = Win_strength, z = rho_test)) +
  geom_contour_filled() + xlab("Win_sparsity") + ylab("Win_strength") +
  NULL

# Visualize the results of logistic time series reservoir ---------------------------------- #
# Patterns in the predicted values
erc_df_train <- data.frame(time = 1:length(py$logis_erc$train_predicted),
                           obs = py$logis_erc$train_true[(py$logis_erc$washout+1):length(py$logis_erc$train_true)],
                           pred = py$logis_erc$train_predicted)
erc_df_test <- data.frame(time = 1:length(py$logis_erc$test_true),
                          obs = py$logis_erc$test_true,
                          pred = py$logis_erc$test_predicted)
erc_df_melt <- pivot_longer(erc_df_test, cols = -1, names_to = "variable", values_to = "value")

log_erc_g1 <- ggplot(erc_df_test, aes(x = obs, y = pred)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("Observed") + ylab("Predicted") + ggtitle("Logistic reservoir\n(Time series)")
log_erc_g2 <- ggplot(erc_df_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("Logistic reservoir\n(Time series)") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value")

# Parameter search
log_erc_g3 <- ggplot(logis_erc_df1, aes(x = num_nodes, y = n_neighbors, z = test_pred)) +
  geom_contour_filled() + xlab("Embedding dimension (E)") + ylab("N of neighbors") +
  NULL
log_erc_g4 <- ggplot(logis_erc_df2, aes(x = Win_sparsity, y = Win_strength, z = test_pred)) +
  geom_contour_filled() + xlab("Win_sparsity") + ylab("Win_strength") +
  NULL


# <----------------------------------------------------> #
#                      Save figures
# <----------------------------------------------------> #
log_eqn_g_all1 <- plot_grid(log_eqn_g1_1, log_eqn_g1_2, log_eqn_g2, rel_widths = c(1,1,2), ncol = 3)
log_eqn_g_all2 <- plot_grid(log_eqn_g3, log_eqn_g4, log_eqn_g5, rel_widths = c(1.2,1.2,1), ncol = 3)
log_eqn_g_all <- plot_grid(log_eqn_g_all1, log_eqn_g_all2, nrow = 2)

log_erc_g_all1 <- plot_grid(log_erc_g1, log_erc_g2, rel_widths = c(1,2), ncol = 2)
log_erc_g_all2 <- plot_grid(log_erc_g3, log_erc_g4, ncol = 2)
log_erc_g_all <- plot_grid(log_erc_g_all1, log_erc_g_all2, nrow = 2)

# Save PDF
pdf(sprintf("%s/Fig_LogisticEquation_LorenzPred.pdf", fig_output_dir), width = 14, height = 8)
log_eqn_g_all; dev.off()
pdf(sprintf("%s/Fig_ERC_LorenzPred.pdf", fig_output_dir), width = 14, height = 8)
log_erc_g_all; dev.off()

# Save listed objects
logistic_eqn_glist <- list(log_eqn_g1_1, log_eqn_g1_2, log_eqn_g2, log_eqn_g3, log_eqn_g4, log_eqn_g5)
logistic_erc_glist <- list(log_erc_g1, log_erc_g2, log_erc_g3, log_erc_g4)
saveRDS(logistic_eqn_glist, sprintf("%s/Figs_ERC_LogisticEQN_LorenzPred.obj", obj_output_dir))
saveRDS(logistic_erc_glist, sprintf("%s/Figs_ERC_LogisticERC_LorenzPred.obj", obj_output_dir))

