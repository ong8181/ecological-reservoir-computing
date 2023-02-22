####
#### Figures: ERC memory capacity
####

# Load library
library(reticulate); packageVersion("reticulate") # 1.26, 2022.11.17
library(tidyverse); packageVersion("tidyverse") # 1.3.2, 2022.11.17
library(cowplot); packageVersion("cowplot") # 1.1.1
library(rEDM); packageVersion("rEDM") # 0.7.5, 2021.8.12
options('tibble.print_max' = 1000)
theme_set(theme_cowplot())

# Create figure output directory
fig_output_dir <- "0_RawFigs"
obj_output_dir <- "0_FigObj"

# Load output data
erc_df_out1 = read.csv("../08_ERC_MemoryOut/ERC_BestTaxa_Memory_df.csv") # Forgotten curve
erc_df_out2 = read.csv("../08_ERC_MemoryOut/ERC_Memory_AllSp_AllDelayMemory.csv") # Memory capacity of all spp.
erc_df_out3 = read.csv("../08_ERC_MemoryOut/MultiERC_Memory_runif_ts.csv") # Forgotten curve of multi-network


# <----------------------------------------------------> #
#  Visualize Forgotten curve and Memory Capacity
# <----------------------------------------------------> #
erc_df_out1$method <- "Single-species"
erc_df_out3$method <- "Multi-species"
erc_forget_1 <- ggplot(erc_df_out1, aes(x = delay, y = test_pred^2)) +
  geom_point() + geom_line() + xlab("Time step") + ylab(expression(R^2))
erc_mc <- erc_df_out2 %>% 
  group_by(reservoir_var) %>%
  summarize(r2 = sum(test_pred^2), num_nodes = mean(num_nodes)) %>% 
  ggplot(aes(x = num_nodes, y = r2)) +
  geom_point() + xlab("Best embedding dimension (E)") + ylab("Memory capacity")
erc_forget_2 <- ggplot(erc_df_out3, aes(x = delay, y = test_pred^2)) +
  geom_point() + geom_line() + xlab("Time step") + ylab(expression(R^2))

erc_df_comb <- rbind(erc_df_out1[,c("delay", "test_pred", "method")],
                     erc_df_out3[,c("delay", "test_pred", "method")])
erc_forget_comb <- ggplot(erc_df_comb, aes(x = delay, y = test_pred^2, color = method)) +
  geom_point() + geom_line() + scale_color_manual(values = c("red3", "black")) +
  xlab("Time step") + ylab(expression(R^2)) + ylim(0, 1)
erc_mc_all <- plot_grid(erc_forget_comb, erc_mc, ncol = 1, align = "hv", axis = "lrbt")


# <----------------------------------------------------> #
#                      Save figures
# <----------------------------------------------------> #
pdf(sprintf("%s/Fig_MemoryCapacity_ERC.pdf", fig_output_dir), width = 8, height = 8)
erc_mc_all; dev.off()

erc_mc_glist <- list(erc_forget_comb, erc_mc)
saveRDS(erc_mc_glist, sprintf("%s/Figs_ERC_MemoryCapacity.obj", obj_output_dir))


