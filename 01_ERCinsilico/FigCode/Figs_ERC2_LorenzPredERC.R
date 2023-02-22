####
#### Figures: ERC Lorenz prediction
####

# Load library
library(reticulate); packageVersion("reticulate") # 1.26, 2022.11.17
library(tidyverse); packageVersion("tidyverse") # 1.3.2, 2022.11.17
library(cowplot); packageVersion("cowplot") # 1.1.1
library(rEDM); packageVersion("rEDM") # 0.7.5
library(ggsci); packageVersion("ggsci") # 2.9
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

# Ecological reservoir
# ! these files are heavy and not included in the Github repository ! #
# Please contact ong8181@gmail.com if necessary
# Or please reproduce them executing the codes
erc_single = joblib.load("04_ERC_LorenzPredOut/ERC_SinglePredLorenz.jb")
erc_multi = joblib.load("04_ERC_LorenzPredOut/ERC_MultiPredLorenz.jb")
exit
#::::::::::::::::::::::::::::::::::::::::::::::::::#
#::::::::::::::::::::::::::::::::::::::::::::::::::#
# Reset working directory
setwd(my_dir)

# Load ERC results
erc_df_out1 = read.csv("../04_ERC_LorenzPredOut/ERC_output_df1.csv")
erc_df_out2 = read.csv("../04_ERC_LorenzPredOut/ERC_output_df2.csv")
erc_df_out3 = read.csv("../04_ERC_LorenzPredOut/ERC_output_df3.csv")
erc_multi_df = read.csv("../04_ERC_LorenzPredOut/MultiERC_bestE.csv")
erc_single_df = read.csv("../04_ERC_LorenzPredOut/SingleERC_bestE.csv")

# Visualize Random Reservoir ---------------------------------- #
erc_df_train <- data.frame(time = 1:length(py$erc_multi$train_predicted),
                           obs = py$erc_multi$train_true[(py$erc_multi$washout+1):length(py$erc_multi$train_true)],
                           pred = py$erc_multi$train_predicted)
erc_df_test <- data.frame(time = 1:length(py$erc_multi$test_true),
                          obs = py$erc_multi$test_true,
                          pred = py$erc_multi$test_predicted)
erc_df_melt <- pivot_longer(erc_df_test, cols = -1, names_to = "variable", values_to = "value")

erc_g1 <- ggplot(erc_df_test, aes(x = obs, y = pred)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("Observed") + ylab("Predicted") + ggtitle("Ecological reservoir (Multinetwork)")
erc_g2_1 <- ggplot(erc_df_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() +
  #ggtitle("Ecological reservoir (Multinetwork)") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value")
erc_g2_2 <- ggplot(erc_multi_df, aes(x = total_nodes, y = test_pred)) +
  geom_point(size = 1) + geom_line() +
  ylim(0.8, 1) + scale_x_log10() +
  xlab("Total reservoir size") + ylab("Correlation coefficient") +
  NULL
erc_g2_3 <- ggplot(erc_multi_df, aes(x = X + 1, y = test_pred)) +
  geom_point(size = 1) + geom_line() +
  ylim(0.8, 1) + scale_x_log10() +
  xlab("The number of species multiplexed") + ylab("Correlation coefficient") +
  NULL

erc_g3_1 <- ggplot(erc_df_out1, aes(x = num_nodes, y = test_pred)) +
  geom_point() + geom_line() + ylim(0.8, 1) + scale_x_log10(limits = c(1, 101)) + 
  xlab("Reservoir size") + ylab("Correlation coefficient") + ggtitle("Prok_Taxa00004")
erc_g3_2 <- ggplot(erc_df_out2, aes(x = num_nodes, y = test_pred)) +
  geom_point() + geom_line() + ylim(0.8, 1) + scale_x_log10(limits = c(1, 101)) + 
  xlab("Reservoir size") + ylab("Correlation coefficient") + ggtitle("Prok_Taxa00005")
erc_g3_3 <- ggplot(erc_df_out3, aes(x = num_nodes, y = test_pred)) +
  geom_point() + geom_line() + ylim(0.8, 1) + scale_x_log10(limits = c(1, 101)) + 
  xlab("Reservoir size") + ylab("Correlation coefficient") + ggtitle("Prok_Taxa00006")


# <----------------------------------------------------> #
#                      Save figures
# <----------------------------------------------------> #
erc_g1_all <- plot_grid(erc_g2_3, erc_g2_1, rel_widths = c(1,2), ncol = 2, align = "hv")
erc_g3_all <- plot_grid(erc_g3_1, erc_g3_2, erc_g3_3, ncol = 3)

ggsave(sprintf("%s/Fig_LorenzPred_ERC_Multi.pdf", fig_output_dir),
       plot = erc_g1_all, width = 12, height = 4)
ggsave(sprintf("%s/Fig_LorenzPred_ERC_Parms.pdf", fig_output_dir),
       plot = erc_g3_all, width = 14, height = 8)

# Save listed objects
erc_lorenzpred_glist <- list(erc_g1, erc_g2_1, erc_g2_2, erc_g2_3, erc_g3_1, erc_g3_2, erc_g3_3)
saveRDS(erc_lorenzpred_glist, sprintf("%s/Figs_ERC_multiERC_LorenzPred.obj", obj_output_dir))
