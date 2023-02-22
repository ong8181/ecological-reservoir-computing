####
#### Figures: Echo State Property
####

# Load library
library(reticulate); packageVersion("reticulate") # 1.26, 2022.11.17
library(tidyverse); packageVersion("tidyverse") # 1.3.2, 2022.11.17
library(cowplot); packageVersion("cowplot") # 1.1.1
library(RColorBrewer); packageVersion("RColorBrewer") # 1.1.2, 2021.7.23
options('tibble.print_max' = 20)
theme_set(theme_cowplot())
get_palette <- colorRampPalette(brewer.pal(8, "Paired"))
palette_manual1 <- get_palette(47)
palette_manual2 <- get_palette(500)

# <----------------------------------------------------> #
#                       Load data
# <----------------------------------------------------> #
# Change workind directory and start python
fig_output_dir <- "0_RawFigs"
obj_output_dir <- "0_FigObj"

# Two specific species (100 random starting points)
# Load results
fish_esp = read_csv("../11_ESPindex_TSpredOut/ESPindex_Fish_PredLorenz.csv")
prok_esp = read_csv("../11_ESPindex_TSpredOut/ESPindex_Prok_PredLorenz.csv")
# Load original data base for species name
ecol_ts = read_csv('../data/edna_asv_table_prok_all.csv')
# All species (10 random starting points for each species)
n_rep <- 5
prok_names <- readRDS("../11_ESPindex_TSpredOut/prok_var_top500.obj")
fish_names <- readRDS("../11_ESPindex_TSpredOut/fish_var_top47.obj")

# ! these files are heavy and not included in the Github repository ! #
# Please contact ong8181@gmail.com if necessary
# Or please reproduce them executing the codes
fish_all = read_csv("../11_ESPindex_TSpredOut/ESPindex_Fish_RandomTS_top47_rep5.csv")
prok_all = read_csv("../11_ESPindex_TSpredOut/ESPindex_Prok_RandomTS_top500_rep5.csv")

# Extract subset of the data to reduce data size
tail(rowSums(fish_all[,2:(ncol(fish_all)-1)]))
tail(rowSums(prok_all[,2:(ncol(prok_all)-1)]))


# Visualize ESP of a particular species
fish_long <- pivot_longer(fish_esp[,-1], cols = -state_id, names_to = "thread", values_to = "value")
prok_long <- pivot_longer(prok_esp[,-1], cols = -state_id, names_to = "thread", values_to = "value")

g1 <- ggplot(fish_long, aes(x = state_id, y = value, color = thread)) +
  geom_line() + scale_color_viridis_d() + theme(legend.position = "none") +
  xlim(0,30) + ggtitle("ESP of fish reservoir")
g2 <- ggplot(prok_long, aes(x = state_id, y = value, color = thread)) +
  geom_line() + scale_color_viridis_d() + theme(legend.position = "none") +
  xlim(0,30) + ggtitle("ESP of prokaryote reservoir")


# Visualize ESP of many species
fish_long2 <- pivot_longer(fish_all[,-1], cols = -state_id, names_to = "thread", values_to = "value")
prok_long2 <- pivot_longer(prok_all[,-1], cols = -state_id, names_to = "thread", values_to = "value")

# Add species name and thread
fish_nrow = dim(fish_all)[1]; prok_nrow = dim(prok_all)[1]
fish_long2$species <- rep(fish_names, times = fish_nrow, each = n_rep)
prok_long2$species <- rep(prok_names, times = prok_nrow, each = n_rep)

g3 <- ggplot(fish_long2, aes(x = state_id, y = value, color = species, group = thread)) +
  geom_line(alpha = 0.5) +
  scale_color_manual(values = palette_manual1) +
  scale_x_log10() +
  theme(legend.position = "none") +
  ggtitle("ESP of fish reservoir") +
  xlab("Time step") + ylab("State difference") +
  NULL
g4 <- ggplot(prok_long2, aes(x = state_id, y = value, color = species, group = thread)) +
  geom_line(alpha = 0.5) +
  scale_color_manual(values = palette_manual2) +
  scale_x_log10() +
  theme(legend.position = "none") +
  ggtitle("ESP of prokaryote reservoir") +
  xlab("Time step") + ylab("State difference") +
  NULL
g5 <- prok_long2 %>% filter(species %in% prok_names[1:30]) %>%
  ggplot(aes(x = state_id, y = value, color = species, group = thread)) +
  geom_line(alpha = 0.5) +
  scale_color_manual(values =  get_palette(30)) +
  scale_x_log10() +
  theme(legend.position = "none") +
  ggtitle("ESP of prokaryote reservoir") +
  xlab("Time step") + ylab("State difference") +
  NULL



# <----------------------------------------------------> #
#                      Save figures
# <----------------------------------------------------> #
esp_all <- plot_grid(g1, g2, ncol = 2)
pdf(sprintf("%s/Fig_ESP_ERC1.pdf", fig_output_dir), width = 8, height = 4)
esp_all; dev.off()

esp_all2 <- plot_grid(g3, g4, ncol = 2)
pdf(sprintf("%s/Fig_ESP_ERC2_All.pdf", fig_output_dir), width = 8, height = 4)
esp_all2; dev.off()
ggsave(sprintf("%s/Figs_ESP2_all.jpg", obj_output_dir), plot = esp_all2, dpi = 300, width = 8, height = 4)
pdf(sprintf("%s/Fig_ESP_ERC2_sub.pdf", fig_output_dir), width = 4, height = 4)
g5; dev.off()

# Save listed objects
esp_glist1 <- list(g1, g2)
saveRDS(esp_glist1, sprintf("%s/Figs_ESP1.obj", obj_output_dir))
esp_glist2 <- list(g3, g4)
saveRDS(esp_glist2, sprintf("%s/Figs_ESP2_all.obj", obj_output_dir))

