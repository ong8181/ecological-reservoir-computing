####
#### Figures: ESN Mackey-Glass
####

# Load library
library(reticulate); packageVersion("reticulate") # 1.26, 2022.11.17
library(tidyverse); packageVersion("tidyverse") # 1.3.2, 2022.11.17
library(cowplot); packageVersion("cowplot") # 1.1.1
library(rEDM); packageVersion("rEDM") # 0.7.5
options('tibble.print_max' = 1000)
theme_set(theme_cowplot())

# <----------------------------------------------------> #
#                       Load data
# <----------------------------------------------------> #
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
# ! these files are heavy and not included in the Github repository ! #
# Please contact ong8181@gmail.com if necessary
# Or please reproduce them executing the codes
esn_mackey = joblib.load("09_ESN_MackeyGlassOut/ESN_MackeyGlass.jb")
exit
#::::::::::::::::::::::::::::::::::::::::::::::::::#
#::::::::::::::::::::::::::::::::::::::::::::::::::#
# Reset working directory
setwd(my_dir)

# <----------------------------------------------------> #
#     Visualize Forgotten curve and Memory Capacity
# <----------------------------------------------------> #
# Patterns in the predicted values
esn_df_train <- data.frame(time = 1:length(py$esn_mackey$train_predicted),
                           obs = py$esn_mackey$train_true[(py$esn_mackey$washout+2):(length(py$esn_mackey$train_true))],
                           pred = py$esn_mackey$train_predicted)
esn_df_test <- data.frame(time = 1:length(py$esn_mackey$test_true),
                          obs = py$esn_mackey$test_true,
                          pred = py$esn_mackey$test_predicted)
esn_df_test_lag <- data.frame(time = 1:length(py$esn_mackey$test_true),
                              obs = dplyr::lag(esn_df_test$obs, n = 17),
                              pred = dplyr::lag(esn_df_test$pred, n = 17))
esn_df_melt <- pivot_longer(esn_df_test, cols = -1, names_to = "variable", values_to = "value")
esn_df_melt_lag <- pivot_longer(esn_df_test_lag, cols = -1, names_to = "variable", values_to = "value")
esn_df_melt_lag$unlagged_value <- esn_df_melt$value

mackey_g1 <- ggplot(esn_df_test, aes(x = obs, y = pred)) +
  geom_point() + geom_abline(intercept = 0, slope = 1, linetype = 2) +
  xlab("Observed") + ylab("Predicted") + ggtitle("Macky-Glass equation")
mackey_g2 <- ggplot(esn_df_melt, aes(x = time, y = value, color = variable, linetype = variable, alpha = variable)) +
  geom_line() + geom_point() + ggtitle("Macky-Glass equation") +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("Time") + ylab("Value")
mackey_g3 <- ggplot(esn_df_melt_lag, aes(x = unlagged_value, y = value,
                    color = variable, linetype = variable, alpha = variable)) + geom_point() +
  scale_color_manual(values = c("black", "red3")) +
  scale_alpha_manual(values = c(1, 0.8)) +
  scale_linetype_manual(values = c(1, 1)) +
  xlab("t") + ylab("t-17")


# <----------------------------------------------------> #
#                      Save figures
# <----------------------------------------------------> #
mackey_all <- plot_grid(mackey_g2,
                        plot_grid(mackey_g1, mackey_g3, rel_widths = c(1,1.2)),
                        ncol = 1)
pdf(sprintf("%s/Fig_MackeyGlass_ESN.pdf", fig_output_dir), width = 10, height = 8)
mackey_all; dev.off()

# Save listed objects
mackey_esn_glist <- list(mackey_g1, mackey_g2, mackey_g3)
saveRDS(mackey_esn_glist, sprintf("%s/Figs_ESN_MackeyGlass.obj", obj_output_dir))
