####
#### Figures: ERC Mackey-Glass
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

# Logistic equation
erc_mackey_taxa1 = joblib.load("10_ERC_MackeyGlassOut/ERCMackeyGlass_SingleBest.jb")
exit
#::::::::::::::::::::::::::::::::::::::::::::::::::#
#::::::::::::::::::::::::::::::::::::::::::::::::::#
# Reset working directory
setwd(my_dir)


# <----------------------------------------------------> #
#     Visualize Forgotten curve and Memory Capacity
# <----------------------------------------------------> #
# Patterns in the predicted values
erc_df_train <- data.frame(time = 1:length(py$erc_mackey_taxa1$train_predicted),
                           obs = py$erc_mackey_taxa1$train_true[(py$erc_mackey_taxa1$washout+2):(length(py$erc_mackey_taxa1$train_true))],
                           pred = py$erc_mackey_taxa1$train_predicted)
erc_df_test <- data.frame(time = 1:length(py$erc_mackey_taxa1$test_true),
                          obs = py$erc_mackey_taxa1$test_true,
                          pred = py$erc_mackey_taxa1$test_predicted)
erc_df_test_lag <- data.frame(time = 1:length(py$erc_mackey_taxa1$test_true),
                          obs = dplyr::lag(erc_df_test$obs, n = 17),
                          pred = dplyr::lag(erc_df_test$pred, n = 17))
erc_df_melt <- pivot_longer(erc_df_test, cols = -1, names_to = "variable", values_to = "value")
erc_df_melt_lag <- pivot_longer(erc_df_test_lag, cols = -1, names_to = "variable", values_to = "value")
erc_df_melt_lag$unlagged_value <- erc_df_melt$value

mackey_g1 <- ggplot(erc_df_test, aes(x = obs, y = pred)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("Observed") + ylab("Predicted")
mackey_g2 <- ggplot(erc_df_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("Macky-Glass equation (ERC)") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value")
mackey_g3 <- ggplot(erc_df_melt_lag, aes(x = unlagged_value, y = value,
                     color = variable, linetype = variable, alpha = variable)) +
  geom_point() +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("t") + ylab("t-17")


# <----------------------------------------------------> #
#                     Save figures
# <----------------------------------------------------> #
mackey_all <- plot_grid(mackey_g2,
                        plot_grid(mackey_g1, mackey_g3, rel_widths = c(1,1.2)),
                        ncol = 1)
pdf(sprintf("%s/Fig_MackeyGlass_ERC.pdf", fig_output_dir), width = 10, height = 8)
mackey_all; dev.off()

mackey_erc_glist <- list(mackey_g1, mackey_g2, mackey_g3)
saveRDS(mackey_erc_glist, sprintf("%s/Figs_ERC_MackeyGlass.obj", obj_output_dir))

