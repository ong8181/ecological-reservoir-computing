####
#### Figures: ESN NARMA emulation
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
rand_esn = joblib.load("05_ESN_NARMAEmlOut/ESN_Eml_narma10.jb")
rand_esn02 = joblib.load("05_ESN_NARMAEmlOut/ESN_Eml_narma02.jb")
rand_esn03 = joblib.load("05_ESN_NARMAEmlOut/ESN_Eml_narma03.jb")
rand_esn04 = joblib.load("05_ESN_NARMAEmlOut/ESN_Eml_narma04.jb")
rand_esn05 = joblib.load("05_ESN_NARMAEmlOut/ESN_Eml_narma05.jb")
rand_esn10 = joblib.load("05_ESN_NARMAEmlOut/ESN_Eml_narma10.jb")
exit
#::::::::::::::::::::::::::::::::::::::::::::::::::#
#::::::::::::::::::::::::::::::::::::::::::::::::::#
# Reset working directory
setwd(my_dir)

# Load output data
esn_df_out1 = read.csv("../05_ESN_NARMAEmlOut/ESN_narma10_df1.csv")
esn_df_out2 = read.csv("../05_ESN_NARMAEmlOut/ESN_narma10_df2.csv")
esn_df_out3 = read.csv("../05_ESN_NARMAEmlOut/ESN_narma10_df3.csv")
esn_df_out4 = read.csv("../05_ESN_NARMAEmlOut/ESN_narma10_df4.csv")
# Load NARMA results
esn_df_narma02 <- data.frame(time = 1:length(py$rand_esn02$test_true),
                             NARMA02 = py$rand_esn02$test_true,
                             Emulated = py$rand_esn02$test_predicted)
esn_df_narma03 <- data.frame(time = 1:length(py$rand_esn03$test_true),
                             NARMA03 = py$rand_esn03$test_true,
                             Emulated = py$rand_esn03$test_predicted)
esn_df_narma04 <- data.frame(time = 1:length(py$rand_esn04$test_true),
                             NARMA04 = py$rand_esn04$test_true,
                             Emulated = py$rand_esn04$test_predicted)
esn_df_narma05 <- data.frame(time = 1:length(py$rand_esn05$test_true),
                             NARMA05 = py$rand_esn05$test_true,
                             Emulated = py$rand_esn05$test_predicted)
esn_df_narma10 <- data.frame(time = 1:length(py$rand_esn$test_true),
                          NARMA10 = py$rand_esn$test_true,
                          Emulated = py$rand_esn$test_predicted)

# <----------------------------------------------------> #
#            Visualize Logistic Reservoir
# <----------------------------------------------------> #
esn_df_train <- data.frame(time = 1:length(py$rand_esn$train_predicted),
                           NARMA10 = py$rand_esn$train_true[(py$rand_esn$washout+1):length(py$rand_esn$train_true)],
                           Emulated = py$rand_esn$train_predicted)
esn_df_test <- data.frame(time = 1:length(py$rand_esn$test_true),
                          NARMA10 = py$rand_esn$test_true,
                          Emulated = py$rand_esn$test_predicted)
esn_df_melt <- pivot_longer(esn_df_test, cols = -1, names_to = "variable", values_to = "value")
esn_df_melt$variable <- factor(esn_df_melt$variable, levels = c("NARMA10", "Emulated"))

esn_g1 <- ggplot(esn_df_test, aes(x = NARMA10, y = Emulated)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("NARMA10") + ylab("Emulated") + ggtitle("Echo State Network")
esn_g2 <- ggplot(esn_df_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("Echo State Network") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value") + xlim(0,200)

esn_g3 <- ggplot(esn_df_out1, aes(x = num_nodes, y = NMSE_test)) +
  geom_point() + geom_line() + ylim(0, 0.15) +
  scale_x_log10(limits = c(10, 1001)) + 
  xlab("Reservoir size") + ylab("NMSE") + ggtitle("Echo State Network")

max(esn_df_out2$NMSE_test, esn_df_out3$NMSE_test, esn_df_out4$NMSE_test)
ymin1 <- 0; ymax1 <- 0.3
esn_g3_1 <- ggplot(esn_df_out2, aes(x = alpha, y = NMSE_test)) +
  geom_point() + geom_line() + ylim(ymin1, ymax1) + 
  xlab(expression(alpha)) + ylab("NMSE") + ggtitle("Echo State Network")
esn_g3_2 <- ggplot(esn_df_out3, aes(x = leak_rate, y = NMSE_test)) +
  geom_point() + geom_line() + ylim(ymin1, ymax1) + 
  xlab("Leak rate") + ylab("NMSE") + ggtitle("Echo State Network")
esn_g3_3 <- esn_df_out4 %>% group_by(Win_sparsity, Win_strength) %>% 
  summarize(Win_sparsity = Win_sparsity, Win_strength = Win_strength, NMSE_test = mean(NMSE_test)) %>% 
  ggplot(aes(x = Win_sparsity, y = Win_strength, z = NMSE_test)) +
  geom_contour_filled() + xlab("Win_sparsity") + ylab("Win_strength") +
  NULL
esn_g3_4 <- esn_df_out4 %>% group_by(Win_sparsity, W_sparsity) %>% 
  summarize(Win_sparsity = Win_sparsity, W_sparsity = W_sparsity, NMSE_test = mean(NMSE_test)) %>% 
  ggplot(aes(x = Win_sparsity, y = W_sparsity, z = NMSE_test)) +
  geom_contour_filled() + xlab("Win_sparsity") + ylab("W_sparsity") +
  NULL
esn_g3_5 <- esn_df_out4 %>% group_by(W_sparsity, Win_strength) %>% 
  summarize(W_sparsity = W_sparsity, Win_strength = Win_strength, NMSE_test = mean(NMSE_test)) %>% 
  ggplot(aes(x = W_sparsity, y = Win_strength, z = NMSE_test)) +
  geom_contour_filled() + xlab("W_sparsity") + ylab("Win_strength") +
  NULL


# <----------------------------------------------------> #
#                      Save figures
# <----------------------------------------------------> #
esn_g_all1 <- plot_grid(esn_g1, esn_g2, rel_widths = c(1,2), ncol = 2)
esn_g_all2 <- plot_grid(esn_g3, esn_g3_1, esn_g3_2, esn_g3_3, esn_g3_4, esn_g3_5, ncol = 3)

pdf(sprintf("%s/Fig_NARMA_ESN1.pdf", fig_output_dir), width = 12, height = 4)
esn_g_all1; dev.off()
pdf(sprintf("%s/Fig_NARMA_ESN2.pdf", fig_output_dir), width = 12, height = 8)
esn_g_all2; dev.off()

# Save listed objects
esn_narma_glist <- list(esn_g1, esn_g2, esn_g3, esn_g3_1, esn_g3_2, esn_g3_3, esn_g3_4)
esn_narma_dfs <- list(esn_df_narma02, esn_df_narma03, esn_df_narma04, esn_df_narma05, esn_df_narma10)
saveRDS(esn_narma_glist, sprintf("%s/Figs_ESN_NARMA.obj", obj_output_dir))
saveRDS(esn_narma_dfs, sprintf("%s/ESN_NARMA_results.obj", obj_output_dir))


