####
#### Figures: ESN Memory capacity
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

# Load output data
esn_df_out1 = read.csv("../07_ESN_MemoryOut/ESN_Memory_df.csv")
esn_df_out2 = read.csv("../07_ESN_MemoryOut/ESN_MemoryCapacity.csv")


# <----------------------------------------------------> #
#     Visualize Forgotten curve and Memory Capacity
# <----------------------------------------------------> #
esn_forget <- ggplot(esn_df_out1, aes(x = delay, y = test_pred^2)) +
  geom_point() + geom_line() + xlab("Time step") + ylab(expression(R^2))
esn_mc <- ggplot(esn_df_out2, aes(x = num_nodes, y = memory_capacity, group = factor(alpha), color = factor(alpha))) +
  geom_point() + geom_line() + xlab("Reservoir size") + ylab("Memory capacity") + scale_color_viridis_d(name = expression(alpha))


# <----------------------------------------------------> #
#                      Save figures
# <----------------------------------------------------> #
esn_mc_all <- plot_grid(esn_forget, esn_mc, ncol = 1, align = "hv", axis = "lrbt")
pdf(sprintf("%s/Fig_MemoryCapacity_ESN.pdf", fig_output_dir), width = 8, height = 8)
esn_mc_all; dev.off()

# Save listed objects
esn_mc_glist <- list(esn_forget, esn_mc)
saveRDS(esn_mc_glist, sprintf("%s/Figs_ESN_MemoryCapacity.obj", obj_output_dir))

