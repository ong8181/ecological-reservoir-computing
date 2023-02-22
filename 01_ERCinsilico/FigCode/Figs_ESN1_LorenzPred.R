####
#### Figures: ESN Lorenz prediction
####

# Load library
library(reticulate); packageVersion("reticulate") # 1.26, 2022.11.17
library(tidyverse); packageVersion("tidyverse") # 1.3.2, 2022.11.17
library(cowplot); packageVersion("cowplot") # 1.1.1
library(rEDM); packageVersion("rEDM") # 0.7.5
library(ggsci); packageVersion("ggsci") # 2.9
theme_set(theme_cowplot())
options('tibble.print_max' = 1000)

# Create figure output directory
fig_output_dir <- "0_RawFigs"
obj_output_dir <- "0_FigObj"

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

# Echo State Network
# ! these files are heavy and not included in the Github repository ! #
# Please contact ong8181@gmail.com if necessary
# Or please reproduce them executing the codes
rand_esn = joblib.load("01_ESN_LorenzPredOut/ESN_PredLorenz.jb")
exit
#::::::::::::::::::::::::::::::::::::::::::::::::::#
#::::::::::::::::::::::::::::::::::::::::::::::::::#
# Reset working directory
setwd(my_dir)

# <----------------------------------------------------> #
#                  Load ESN results
# <----------------------------------------------------> #
esn_df_out1 = read.csv("../01_ESN_LorenzPredOut/ESN_output_df1.csv")
esn_df_out2 = read.csv("../01_ESN_LorenzPredOut/ESN_output_df2.csv")
esn_df_out3 = read.csv("../01_ESN_LorenzPredOut/ESN_output_df3.csv")
esn_mn_E003 = read.csv("../01_ESN_LorenzPredOut/MultiNetowrk_E003.csv")
esn_mn_E005 = read.csv("../01_ESN_LorenzPredOut/MultiNetowrk_E005.csv")
esn_mn_E010 = read.csv("../01_ESN_LorenzPredOut/MultiNetowrk_E010.csv")
esn_mn_E020 = read.csv("../01_ESN_LorenzPredOut/MultiNetowrk_E020.csv")
esn_mn_E030 = read.csv("../01_ESN_LorenzPredOut/MultiNetowrk_E030.csv")
esn_mn_E040 = read.csv("../01_ESN_LorenzPredOut/MultiNetowrk_E040.csv")
esn_mn_E050 = read.csv("../01_ESN_LorenzPredOut/MultiNetowrk_E050.csv")
esn_mn_E100 = read.csv("../01_ESN_LorenzPredOut/MultiNetowrk_E100.csv")
esn_sngl = read.csv("../01_ESN_LorenzPredOut/SingleNetwork_E1000.csv")
esn_mn_all <- rbind(esn_mn_E003, esn_mn_E005, esn_mn_E010, esn_mn_E020,
                    esn_mn_E030, esn_mn_E040, esn_mn_E050, esn_mn_E100)

# Load reservoir time series
lorenz <- read.csv("../data/LorenzTS_tau10.csv")
logis_ts <- read.csv("../data/logistic_Xr2_92Yr2_90.csv")
ecol_ts <- read.csv("../data/edna_asv_table_prok_all.csv")
ecol_ts$time_index <- rep(1:153, 5)
ecol_ts$plot <- c(rep("Plot 1", 153), rep("Plot 2", 153), rep("Plot 3", 153), rep("Plot 4", 153), rep("Plot 5", 153))

# Visualize Lorenz and logistic time series ---------------------------------- #
g_lorenz_x <- ggplot(lorenz, aes(x = time, y = x)) + geom_line(linewidth = 0.3) + xlab("Time") + xlim(0,50)
g_lorenz_y <- ggplot(lorenz, aes(x = time, y = y)) + geom_line(linewidth = 0.3) + xlab("Time") + xlim(0,50)
g_lorenz_z <- ggplot(lorenz, aes(x = time, y = z)) + geom_line(linewidth = 0.3) + xlab("Time") + xlim(0,50)
g_logis_x <- ggplot(logis_ts, aes(x = time, y = x)) + geom_line(linewidth = 0.3) + xlab("Time") + xlim(0,150)
g_logis_y <- ggplot(logis_ts, aes(x = time, y = y)) + geom_line(linewidth = 0.3) + xlab("Time") + xlim(0,150)
g_ecol <- ggplot(ecol_ts, aes(x = time_index, y = Prok_Taxa00005, color = plot)) + geom_line(linewidth = 0.3) +
  xlab("Date index") + ylab("DNA copy numbers") + scale_color_startrek() + xlim(23, 150)
g_logis_xy <- ggplot(logis_ts, aes(x = x, y = y)) + geom_point()



# Visualize Echo State Network ---------------------------------- #
esn_df_train <- data.frame(time = 1:length(py$rand_esn$train_predicted),
                           obs = py$rand_esn$train_true[(py$rand_esn$washout+1):length(py$rand_esn$train_true)],
                           pred = py$rand_esn$train_predicted)
esn_df_test <- data.frame(time = 1:length(py$rand_esn$test_true),
                          obs = py$rand_esn$test_true,
                          pred = py$rand_esn$test_predicted)
esn_df_melt <- pivot_longer(esn_df_test, cols = -1, names_to = "variable", values_to = "value")

esn_g1 <- ggplot(esn_df_test, aes(x = obs, y = pred)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("Observed") + ylab("Predicted") + ggtitle("Random reservoir\n(Equation)")
esn_g2 <- ggplot(esn_df_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("Random reservoir\n(Equation)") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value")

esn_g3 <- ggplot(esn_df_out1, aes(x = num_nodes, y = rho_test)) +
  geom_point() + geom_line() + ylim(0.94, 1) + 
  scale_x_log10(limits = c(10, 1001)) + 
  xlab("Reservoir size") + ylab("Correlation coefficient") + ggtitle("Echo State Network")

esn_g3_1 <- ggplot(esn_df_out2, aes(x = alpha, y = rho_test)) +
  geom_point() + geom_line() + ylim(0.845, 1) +
  xlab(expression(alpha)) + ylab("Correlation coefficient") + ggtitle("Echo State Network")
esn_g3_2 <- esn_df_out3 %>% group_by(Win_sparsity, Win_strength) %>% 
  summarize(Win_sparsity = Win_sparsity, Win_strength = Win_strength, rho_test = mean(rho_test)) %>% 
  ggplot(aes(x = Win_sparsity, y = Win_strength, z = rho_test)) +
  geom_contour_filled() + xlab("Win_sparsity") + ylab("Win_strength") +
  NULL
esn_g3_3 <- esn_df_out3 %>% group_by(Win_sparsity, W_sparsity) %>% 
  summarize(Win_sparsity = Win_sparsity, W_sparsity = W_sparsity, rho_test = mean(rho_test)) %>% 
  ggplot(aes(x = Win_sparsity, y = W_sparsity, z = rho_test)) +
  geom_contour_filled() + xlab("Win_sparsity") + ylab("W_sparsity") +
  NULL
esn_g3_4 <- esn_df_out3 %>% group_by(Win_strength, W_sparsity) %>% 
  summarize(Win_sparsity = Win_strength, W_sparsity = W_sparsity, rho_test = mean(rho_test)) %>% 
  ggplot(aes(x = Win_strength, y = W_sparsity, z = rho_test)) +
  geom_contour_filled() + xlab("Win_strength") + ylab("W_sparsity") +
  NULL


esn_g4_1 <- ggplot(esn_mn_all, aes(x = total_nodes, y = test_pred, color = as.factor(E))) +
  geom_point() + geom_line() + ylim(0.83, 1.0) + scale_color_igv() + scale_x_log10(limits = c(1,11000)) + labs(color = "Reservoir\nsize") +
  xlab("Total number of nodes") + ylab("Correlation coefficient") + ggtitle("Echo State Network") +
  geom_point(data = esn_sngl, aes(x = num_nodes, y = rho_test), color = "black") +
  geom_line(data = esn_sngl, aes(x = num_nodes, y = rho_test), color = "black", linetype = 2)
esn_g4_2 <- ggplot(esn_mn_all, aes(x = n_network, y = test_pred, color = as.factor(E))) +
  geom_point() + geom_line() + ylim(0.83, 1.0) + scale_color_igv() + scale_x_log10(limits = c(1, 300)) + labs(color = "Reservoir\nsize") +
  xlab("Number of networks") + ylab("Correlation coefficient") + ggtitle("Echo State Network")


# <----------------------------------------------------> #
#                      Save figures
# <----------------------------------------------------> #
esn_g4_all <- plot_grid(esn_g4_1, esn_g4_2, ncol = 2)
pdf(sprintf("%s/Fig_LorenzPred_ESN_MultiNetwork.pdf", fig_output_dir), width = 14, height = 6)
esn_g4_all; dev.off()

ts_all_glist <- list(g_lorenz_x, g_lorenz_y, g_lorenz_z, g_logis_x, g_logis_y)
                     #g_ecol, g_fish, g_narma02, g_mackey, g_logis_xy)
esn_all_glist <- list(esn_g1, esn_g2, esn_g3,
                      esn_g3_1, esn_g3_2, esn_g3_3, esn_g3_4,
                      esn_g4_1, esn_g4_2)
saveRDS(ts_all_glist, sprintf("%s/Figs_TimeSeriesExample.obj", obj_output_dir))
saveRDS(esn_all_glist, sprintf("%s/Figs_ESN_LorenzPred.obj", obj_output_dir))
