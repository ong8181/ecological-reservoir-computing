####
#### Figures: ERC NARMA emulation
####

# Load library
library(reticulate); packageVersion("reticulate") # 1.26, 2022.11.17
library(tidyverse); packageVersion("tidyverse") # 1.3.2, 2022.11.17
library(cowplot); packageVersion("cowplot") # 1.1.1
library(rEDM); packageVersion("rEDM") # 0.7.5
options('tibble.print_max' = 1000)
theme_set(theme_cowplot())

# Create figure output directory
fig_output_dir <- "0_RawFigs"
obj_output_dir <- "0_FigObj"
#dir.create("XXX")
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
erc_multi02 = joblib.load("06_ERC_NARMAEmlOut/MultiERC_narma02.jb")
erc_multi03 = joblib.load("06_ERC_NARMAEmlOut/MultiERC_narma03.jb")
erc_multi04 = joblib.load("06_ERC_NARMAEmlOut/MultiERC_narma04.jb")
erc_multi05 = joblib.load("06_ERC_NARMAEmlOut/MultiERC_narma05.jb")
erc_multi10 = joblib.load("06_ERC_NARMAEmlOut/MultiERC_narma10.jb")
exit
#::::::::::::::::::::::::::::::::::::::::::::::::::#
#::::::::::::::::::::::::::::::::::::::::::::::::::#

# Reset working directory
setwd(my_dir)

# Load output data
single_narma02_df = read.csv("../06_ERC_NARMAEmlOut/SingleERC_narma02.csv")
single_narma03_df = read.csv("../06_ERC_NARMAEmlOut/SingleERC_narma03.csv")
single_narma04_df = read.csv("../06_ERC_NARMAEmlOut/SingleERC_narma04.csv")
single_narma05_df = read.csv("../06_ERC_NARMAEmlOut/SingleERC_narma05.csv")
single_narma10_df = read.csv("../06_ERC_NARMAEmlOut/SingleERC_narma10.csv")

multi_narma02_df = read.csv("../06_ERC_NARMAEmlOut/MultiERC_narma02.csv")
multi_narma03_df = read.csv("../06_ERC_NARMAEmlOut/MultiERC_narma03.csv")
multi_narma04_df = read.csv("../06_ERC_NARMAEmlOut/MultiERC_narma04.csv")
multi_narma05_df = read.csv("../06_ERC_NARMAEmlOut/MultiERC_narma05.csv")
multi_narma10_df = read.csv("../06_ERC_NARMAEmlOut/MultiERC_narma10.csv")

# Visualize results of single ERC ---------------------------------- #
single_all <- rbind(single_narma02_df, single_narma03_df, single_narma04_df,
                    single_narma05_df, single_narma10_df)
single_g1 <- ggplot(single_narma02_df, aes(x = num_nodes, y = NMSE_test)) +
  geom_point(alpha = 1) + xlab("Embedding dimension (E)") + ylab("NMSE") +
  ggtitle("Emulation of NARMA02 by ERC") + scale_y_log10(limits = c(0.005, 2))
single_g2 <- ggplot(single_narma03_df, aes(x = num_nodes, y = NMSE_test)) +
  geom_point(alpha = 1) + xlab("Embedding dimension (E)") + ylab("NMSE") +
  ggtitle("Emulation of NARMA03 by ERC") + scale_y_log10(limits = c(0.005, 2))
single_g3 <- ggplot(single_narma04_df, aes(x = num_nodes, y = NMSE_test)) +
  geom_point(alpha = 1) + xlab("Embedding dimension (E)") + ylab("NMSE") +
  ggtitle("Emulation of NARMA04 by ERC") + scale_y_log10(limits = c(0.005, 2))
single_g4 <- ggplot(single_narma05_df, aes(x = num_nodes, y = NMSE_test)) +
  geom_point(alpha = 1) + xlab("Embedding dimension (E)") + ylab("NMSE") +
  ggtitle("Emulation of NARMA05 by ERC") + scale_y_log10(limits = c(0.005, 2))
single_g5 <- ggplot(single_narma10_df, aes(x = num_nodes, y = NMSE_test)) +
  geom_point(alpha = 1) + xlab("Embedding dimension (E)") + ylab("NMSE") +
  ggtitle("Emulation of NARMA10 by ERC") + scale_y_log10(limits = c(0.005, 2))
single_gg <- ggplot(single_all, aes(x = num_nodes, y = NMSE_test, color = target_db_name)) +
  geom_point(alpha = 1) + xlab("Embedding dimension (E)") + ylab("NMSE") +
  scale_color_viridis_d(name = "NARMA") +
  ggtitle("Emulation of NARMA by ERC") + scale_y_log10(limits = c(0.005, 2))

# Visualize resutls of multiple ERC ---------------------------------- #
multi_all <- rbind(multi_narma02_df, multi_narma03_df, multi_narma04_df,
                   multi_narma05_df, multi_narma10_df)
multi_all$cat <- str_sub(multi_all$network_name, start = -7)
multi_g1 <- ggplot(multi_narma02_df, aes(x = total_nodes, y = NMSE_test)) +
  geom_point() + geom_line() + xlab("Total reservoir size") + ylab("NMSE") +
  ggtitle("Emulation of NARMA02 by multi-ERC") + scale_y_log10(limits = c(0.0001, 1))
multi_g2 <- ggplot(multi_narma03_df, aes(x = total_nodes, y = NMSE_test)) +
  geom_point() + geom_line() + xlab("Total reservoir size") + ylab("NMSE") +
  ggtitle("Emulation of NARMA03 by multi-ERC") + scale_y_log10(limits = c(0.0001, 1))
multi_g3 <- ggplot(multi_narma04_df, aes(x = total_nodes, y = NMSE_test)) +
  geom_point() + geom_line() + xlab("Total reservoir size") + ylab("NMSE") +
  ggtitle("Emulation of NARMA04 by multi-ERC") + scale_y_log10(limits = c(0.0001, 1))
multi_g4 <- ggplot(multi_narma05_df, aes(x = total_nodes, y = NMSE_test)) +
  geom_point() + geom_line() + xlab("Total reservoir size") + ylab("NMSE") +
  ggtitle("Emulation of NARMA05 by multi-ERC") + scale_y_log10(limits = c(0.0001, 1))
multi_g5 <- ggplot(multi_narma10_df, aes(x = total_nodes, y = NMSE_test)) +
  geom_point() + geom_line() + xlab("Total reservoir size") + ylab("NMSE") +
  ggtitle("Emulation of NARMA10 by multi-ERC") + scale_y_log10(limits = c(0.0001, 1))
multi_gg <- ggplot(multi_all, aes(x = total_nodes, y = NMSE_test, color = cat)) +
  geom_point(alpha = 1) + geom_line() + xlab("Embedding dimension (E)") + ylab("NMSE") +
  scale_color_viridis_d(name = "NARMA") +
  ggtitle("Emulation of NARMA by ERC") + scale_y_log10(limits = c(0.0001, 2))


# <----------------------------------------------------> #
#  Visualize results of NARMA02 by multiple ERC
# <----------------------------------------------------> #
# NARMA02
narma02_train <- data.frame(time = 1:length(py$erc_multi02$train_predicted),
                           NARMA02 = py$erc_multi02$train_true[(py$erc_multi02$washout+1):length(py$erc_multi02$train_true)],
                           Emulated = py$erc_multi02$train_predicted)
narma02_test <- data.frame(time = 1:length(py$erc_multi02$test_true),
                          NARMA02 = py$erc_multi02$test_true,
                          Emulated = py$erc_multi02$test_predicted)
narma02_melt <- pivot_longer(narma02_test, cols = -1, names_to = "variable", values_to = "value")
narma02_melt$variable <- factor(narma02_melt$variable, levels = c("NARMA02", "Emulated"))

narma02_g1 <- ggplot(narma02_test, aes(x = NARMA02, y = Emulated)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("NARMA02") + ylab("Emulated") + ggtitle("NARMA02 by Multi-ERC")
narma02_g2 <- ggplot(narma02_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("NARMA02 by Multi-ERC") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value") + xlim(0,200)

# NARMA03
narma03_train <- data.frame(time = 1:length(py$erc_multi03$train_predicted),
                            NARMA03 = py$erc_multi03$train_true[(py$erc_multi03$washout+1):length(py$erc_multi03$train_true)],
                            Emulated = py$erc_multi03$train_predicted)
narma03_test <- data.frame(time = 1:length(py$erc_multi03$test_true),
                           NARMA03 = py$erc_multi03$test_true,
                           Emulated = py$erc_multi03$test_predicted)
narma03_melt <- pivot_longer(narma03_test, cols = -1, names_to = "variable", values_to = "value")
narma03_melt$variable <- factor(narma03_melt$variable, levels = c("NARMA03", "Emulated"))

narma03_g1 <- ggplot(narma03_test, aes(x = NARMA03, y = Emulated)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("NARMA03") + ylab("Emulated") + ggtitle("NARMA03 by Multi-ERC")
narma03_g2 <- ggplot(narma03_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("NARMA03 by Multi-ERC") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value") + xlim(0,200)

# NARMA04
narma04_train <- data.frame(time = 1:length(py$erc_multi04$train_predicted),
                            NARMA04 = py$erc_multi04$train_true[(py$erc_multi04$washout+1):length(py$erc_multi04$train_true)],
                            Emulated = py$erc_multi04$train_predicted)
narma04_test <- data.frame(time = 1:length(py$erc_multi04$test_true),
                           NARMA04 = py$erc_multi04$test_true,
                           Emulated = py$erc_multi04$test_predicted)
narma04_melt <- pivot_longer(narma04_test, cols = -1, names_to = "variable", values_to = "value")
narma04_melt$variable <- factor(narma04_melt$variable, levels = c("NARMA04", "Emulated"))

narma04_g1 <- ggplot(narma04_test, aes(x = NARMA04, y = Emulated)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("NARMA04") + ylab("Emulated") + ggtitle("NARMA04 by Multi-ERC")
narma04_g2 <- ggplot(narma04_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("NARMA04 by Multi-ERC") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value") + xlim(0,200)


# NARMA05
narma05_train <- data.frame(time = 1:length(py$erc_multi05$train_predicted),
                            NARMA05 = py$erc_multi05$train_true[(py$erc_multi05$washout+1):length(py$erc_multi05$train_true)],
                            Emulated = py$erc_multi05$train_predicted)
narma05_test <- data.frame(time = 1:length(py$erc_multi05$test_true),
                           NARMA05 = py$erc_multi05$test_true,
                           Emulated = py$erc_multi05$test_predicted)
narma05_melt <- pivot_longer(narma05_test, cols = -1, names_to = "variable", values_to = "value")
narma05_melt$variable <- factor(narma05_melt$variable, levels = c("NARMA05", "Emulated"))

narma05_g1 <- ggplot(narma05_test, aes(x = NARMA05, y = Emulated)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("NARMA05") + ylab("Emulated") + ggtitle("NARMA05 by Multi-ERC")
narma05_g2 <- ggplot(narma05_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("NARMA05 by Multi-ERC") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value") + xlim(0,200)


# NARMA10
narma10_train <- data.frame(time = 1:length(py$erc_multi10$train_predicted),
                            NARMA10 = py$erc_multi10$train_true[(py$erc_multi10$washout+1):length(py$erc_multi10$train_true)],
                            Emulated = py$erc_multi10$train_predicted)
narma10_test <- data.frame(time = 1:length(py$erc_multi10$test_true),
                           NARMA10 = py$erc_multi10$test_true,
                           Emulated = py$erc_multi10$test_predicted)
narma10_melt <- pivot_longer(narma10_test, cols = -1, names_to = "variable", values_to = "value")
narma10_melt$variable <- factor(narma10_melt$variable, levels = c("NARMA10", "Emulated"))

narma10_g1 <- ggplot(narma10_test, aes(x = NARMA10, y = Emulated)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("NARMA10") + ylab("Emulated") + ggtitle("NARMA05 by Multi-ERC")
narma10_g2 <- ggplot(narma10_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("NARMA10 by Multi-ERC") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value") + xlim(0,200)

# Combine figures
single_g_all <- plot_grid(single_g1, single_g2, single_g3, single_g4, single_g5,
                          ncol = 3)
multi_g_all <- plot_grid(multi_g1, multi_g2, multi_g3, multi_g4, multi_g5,
                         ncol = 3)
narma_g_all <- plot_grid(narma02_g1, narma02_g2,
                         narma03_g1, narma03_g2,
                         narma04_g1, narma04_g2,
                         narma05_g1, narma05_g2,
                         narma10_g1, narma10_g2,
                         ncol = 2, rel_widths = c(1,2))


# <----------------------------------------------------> #
#               Compare NARMA and MC
# <----------------------------------------------------> #
# # Load output data
mc_df_out2 = read.csv("../08_ERC_MemoryOut/ERC_Memory_AllSp_AllDelayMemory.csv") # Memory capacity of all spp.
# Calculate memory capacity
mc_df_out2$test_pred_zero <- mc_df_out2$test_pred
mc_df_out2$test_pred_zero[mc_df_out2$test_pred < 0] <- 0

mc_narma02_df <- mc_narma03_df <- mc_narma04_df <- mc_narma05_df <- mc_narma10_df <-
  data.frame(mc_df_out2 %>% group_by(reservoir_var) %>% summarize(mc = sum(test_pred_zero))) %>% rename(species = reservoir_var)
mc_narma02_df$narma_NMSE <- single_narma02_df[match(single_narma02_df$reservoir_var, mc_narma02_df$species), "NMSE_test"]
mc_narma03_df$narma_NMSE <- single_narma03_df[match(single_narma03_df$reservoir_var, mc_narma03_df$species), "NMSE_test"]
mc_narma04_df$narma_NMSE <- single_narma04_df[match(single_narma04_df$reservoir_var, mc_narma04_df$species), "NMSE_test"]
mc_narma05_df$narma_NMSE <- single_narma05_df[match(single_narma05_df$reservoir_var, mc_narma05_df$species), "NMSE_test"]
mc_narma10_df$narma_NMSE <- single_narma10_df[match(single_narma10_df$reservoir_var, mc_narma10_df$species), "NMSE_test"]

mc_narma02 <- ggplot(mc_narma02_df, aes(x = mc, y = narma_NMSE)) + geom_point() +
  xlab("Memory Capacity") + ylab("NMSE") + ggtitle("NARMA02 emulation") +
  scale_y_log10(limits = c(0.001, 1))
mc_narma03 <- ggplot(mc_narma03_df, aes(x = mc, y = narma_NMSE)) + geom_point() +
  xlab("Memory Capacity") + ylab("NMSE") + ggtitle("NARMA03 emulation") +
  scale_y_log10(limits = c(0.065, .5))
mc_narma04 <- ggplot(mc_narma04_df, aes(x = mc, y = narma_NMSE)) + geom_point() +
  xlab("Memory Capacity") + ylab("NMSE") + ggtitle("NARMA04 emulation") +
  scale_y_log10(limits = c(0.065, .5))
mc_narma05 <- ggplot(mc_narma05_df, aes(x = mc, y = narma_NMSE)) + geom_point() +
  xlab("Memory Capacity") + ylab("NMSE") + ggtitle("NARMA05 emulation") +
  scale_y_log10(limits = c(0.065, .5))
mc_narma10 <- ggplot(mc_narma10_df, aes(x = mc, y = narma_NMSE)) + geom_point() +
  xlab("Memory Capacity") + ylab("NMSE") + ggtitle("NARMA10 emulation") +
  scale_y_log10(limits = c(0.065, .5))

mc_narma_all <- plot_grid(mc_narma02, mc_narma03, mc_narma04, mc_narma05, mc_narma10,
                          ncol = 3)


# <----------------------------------------------------> #
#                      Save figures
# <----------------------------------------------------> #
# Save figures
pdf(sprintf("%s/Fig_NARMA_SingleERC.pdf", fig_output_dir), width = 14, height = 8)
single_g_all; dev.off()
pdf(sprintf("%s/Fig_NARMA_MultiERC.pdf", fig_output_dir), width = 14, height = 8)
multi_g_all; dev.off()
pdf(sprintf("%s/Fig_NARMA_All.pdf", fig_output_dir), width = 18, height = 21)
narma_g_all; dev.off()
pdf(sprintf("%s/Fig_NARMA_MC.pdf", fig_output_dir), width = 14, height = 8)
mc_narma_all; dev.off()

# Save listed objects
single_glist <- list(single_g1, single_g2, single_g3, single_g4, single_g5)
multi_glist <- list(multi_g1, multi_g2, multi_g3, multi_g4, multi_g5)
narma_glist <- list(narma02_g1, narma02_g2, narma03_g1, narma03_g2,
                     narma04_g1, narma04_g2, narma05_g1, narma05_g2,
                     narma10_g1, narma10_g2)
mc_narma_glist <- list(mc_narma02, mc_narma03, mc_narma04, mc_narma05, mc_narma10)

saveRDS(single_glist, sprintf("%s/Figs_ERC_NARMA_singleERC.obj", obj_output_dir))
saveRDS(multi_glist, sprintf("%s/Figs_ERC_NARMA_multiERC.obj", obj_output_dir))
saveRDS(narma_glist, sprintf("%s/Figs_ERC_NARMA_multiSummary.obj", obj_output_dir))
saveRDS(mc_narma_glist, sprintf("%s/Figs_ERC_NARMA_MC.obj", obj_output_dir))

