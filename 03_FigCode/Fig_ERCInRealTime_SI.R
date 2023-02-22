####
#### Assemble figures for ERC paper
#### SI for real-time ERC
####

# Load library
library(tidyverse); packageVersion("tidyverse") # 1.3.0, 2020.4.21
library(cowplot); packageVersion("cowplot") # 1.0.0, 2021.3.30
library(ggsci); packageVersion("ggsci") # 2.9, 2021.3.30
library(magick); packageVersion("magick") # 2.6.0, 2020.3.31
library(viridis); packageVersion("viridis") # 0.5.1, 2021.3.30
options('tibble.print_max' = 20)
theme_set(theme_cowplot())


# <----------------------------------------------------> #
#            Specify input/output directory
# <----------------------------------------------------> #
# Create figure output directory
fig_output_dir <- "0_FormattedFigs"
fig_input_dir1 <- "../02_ERCrealtime/v202103_runif_all/02_MeasureMCOut"
fig_input_dir2 <- "../02_ERCrealtime/v202103_lorenz_all/02_PredictionOut"
fig_input_dir3 <- "../02_ERCrealtime/v202103_fishts_all/02_PredictionOut"
fig_input_dir4 <- "../02_ERCrealtime/v202104_growth_rate/01_LoadAllDataOut"
fig_input_dir5 <- "../02_ERCrealtime/v202104_growth_rate/02_LoadAllDataOut"
fig_input_dir7 <- "../02_ERCrealtime/v202105_fishts_all/02_PredictionOut"
dir.create(fig_output_dir)


# <----------------------------------------------------> #
#                   Load figure data
# <----------------------------------------------------> #
# Random uniform input
runif_fig <- readRDS(sprintf("%s/FigObject.obj", fig_input_dir1))
runif_parm <- readRDS(sprintf("%s/ParameterList.obj", fig_input_dir1))
runif_result <- readRDS(sprintf("%s/ReservoirResult.obj", fig_input_dir1))
runif_state <- readRDS(sprintf("%s/ReservoirState.obj", fig_input_dir1))
runif_example <- readRDS(sprintf("%s/ResultExample.obj", fig_input_dir1))

# Lorenz attractor
lorenz_fig <- readRDS(sprintf("%s/FigObject.obj", fig_input_dir2))
lorenz_parm <- readRDS(sprintf("%s/ParameterList.obj", fig_input_dir2))
lorenz_result <- readRDS(sprintf("%s/ReservoirResult.obj", fig_input_dir2))
lorenz_state <- readRDS(sprintf("%s/ReservoirState.obj", fig_input_dir2))
lorenz_example <- readRDS(sprintf("%s/ResultExample.obj", fig_input_dir2))

# Fish catch time-series
fish1_fig <- readRDS(sprintf("%s/FigObject.obj", fig_input_dir3))
fish1_parm <- readRDS(sprintf("%s/ParameterList.obj", fig_input_dir3))
fish1_result <- readRDS(sprintf("%s/ReservoirResult.obj", fig_input_dir3))
fish1_state <- readRDS(sprintf("%s/ReservoirState.obj", fig_input_dir3))
fish1_example <- readRDS(sprintf("%s/ResultExample.obj", fig_input_dir3))

# Fish catch time-series (Maaji)
fish2_fig <- readRDS(sprintf("%s/FigObject.obj", fig_input_dir7))
fish2_parm <- readRDS(sprintf("%s/ParameterList.obj", fig_input_dir7))
fish2_result <- readRDS(sprintf("%s/ReservoirResult.obj", fig_input_dir7))
fish2_state <- readRDS(sprintf("%s/ReservoirState.obj", fig_input_dir7))
fish2_example <- readRDS(sprintf("%s/ResultExample.obj", fig_input_dir7))

# Temperature- and medium-dependence of the growth rate
growth_temp <- readRDS(sprintf("%s/TempDependence.obj", fig_input_dir4))
growth_medium <- readRDS(sprintf("%s/MediumDependence.obj", fig_input_dir5))



# <----------------------------------------------------> #
#                  Format figures
# <----------------------------------------------------> #
# Temperature- and medium-dependence of the growth rate
gr_parm1 <- plot_grid(growth_temp[[2]], growth_temp[[3]], ncol = 2, align = "hv", labels = c("b", "c"))
Fig_TempEffect <- plot_grid(growth_temp[[1]], gr_parm1, ncol = 1, rel_heights = c(1,0.8), labels = c("a", NULL))

gr_parm2 <- plot_grid(growth_medium[[2]], growth_medium[[3]], ncol = 2, align = "hv", labels = c("b", "c"))
Fig_MediumEffect <- plot_grid(growth_medium[[1]], gr_parm2, ncol = 1, rel_heights = c(1,0.8), labels = c("a", NULL))

# Figure: cell count data convert
cell_figs <- readRDS("../02_ERCrealtime/v202103_runif_all/00_RunData/v20210227_runif02/03_CompileDataOut/cell_fig_data.obj")
crop_size <- "960x720+300+300"
# 1920x1440 = 3.6mm x 2.7 mm
# 960x720 = 1.8 mm x 1.35 mm
cell_image1 <- image_read("0_CellImages/Step1_2021-05-25_09-01-34_w_scale.jpg") %>% image_crop(crop_size)
cell_image2 <- image_read("0_CellImages/Step2_2021-05-25_09-01-34.jpg") %>% image_crop(crop_size)
cell_image3 <- image_read("0_CellImages/Step3_2021-05-25_09-01-34_analyzed.jpg") %>% image_crop(crop_size)

gg_cell1 <- ggdraw() + draw_image(image_resize(cell_image1, "x300"))
gg_cell2 <- ggdraw() + draw_image(image_resize(cell_image2, "x300"))
gg_cell3 <- ggdraw() + draw_image(image_resize(cell_image3, "x300"))
Fig_CellImage <- plot_grid(gg_cell1, NULL, gg_cell2, NULL, gg_cell3,
                           ncol = 5, rel_widths = c(1,0.1,1,0.1,1))

Fig_CellCount <- plot_grid(Fig_CellImage,
                           cell_figs[[3]],
                           cell_figs[[4]] + ylab("GAM residuals"),
                           cell_figs[[6]] + ylab("GAM residuals\n (/[cell density + 10])") + geom_hline(yintercept = 0, linetype = 2, color = "red3"),
                           ncol = 1, align = "hv", labels = c("a", "b", "c", "d"))


# Figure labels
legend_label <- c("Temperature (input)", "Temperature (medium)",
                  "Med. nutrient 1", "Med. nutrient 2",
                  "Low nutrient 1", "Low nutrient 2",
                  "High nutrient 1", "High nutrient 2")

# Figure: Reservoir state of Lorenz attractor
a <- 1
lorenz_state2 <- lorenz_state %>%
  mutate(run2 = run2 - 1*a) %>%
  mutate(run3 = run3 - 3*a) %>%
  mutate(run4 = run4 - 4*a) %>%
  mutate(run5 = run5 - 6*a) %>%
  mutate(run6 = run6 - 7*a) %>%
  mutate(temperature = temperature + 3*a) %>%
  mutate(temperature_input = temperature_input + 7*a)
colnames(lorenz_state2)[2:9] <- legend_label
lorenz_state2 <- lorenz_state2 %>% pivot_longer(cols = -time_id) %>% arrange(time_id, name)
Fig_LorenzState0 <- lorenz_state2 %>%
  ggplot(aes(x = time_id, y = value, color = name)) + 
  geom_hline(yintercept = c(0, -1, -3, -4, -6, -7)*a, linetype = 2) +
  geom_line() +
  theme(axis.text.y = element_blank()) +
  scale_color_manual(values = c("darkred", "darkred",
                                "orange2", "orange2",
                                "darkblue", "darkblue", "red4", "red3")) +
  xlab("Time step") +
  NULL

# Figure: Reservoir state of empirical fish catch time series
fish1_state2 <- fish1_state %>% 
  mutate(run2 = run2 - 1*a) %>%
  mutate(run3 = run3 - 3*a) %>%
  mutate(run4 = run4 - 4*a) %>%
  mutate(run5 = run5 - 6*a) %>%
  mutate(run6 = run6 - 7*a) %>%
  mutate(temperature = temperature + 3*a) %>%
  mutate(temperature_input = temperature_input + 7*a)
colnames(fish1_state2)[2:9] <- legend_label
fish1_state2 <- fish1_state2 %>% pivot_longer(cols = -time_id) %>% arrange(time_id, name)
Fig_FishState1 <- fish1_state2 %>%
  ggplot(aes(x = time_id, y = value, color = name)) + 
  geom_hline(yintercept = c(0,-1,-3,-4,-6,-7)*a, linetype = 2) +
  geom_line() +
  theme(axis.text.y = element_blank()) +
  scale_color_manual(values = c("darkred", "darkred",
                                "orange2", "orange2",
                                "darkblue", "darkblue", "red4", "red3")) +
  xlab("Time step") +
  NULL

# Figure: Reservoir state of empirical fish catch time series (Maaji)
fish2_state2 <- fish2_state %>% 
  mutate(run2 = run2 - 1*a) %>%
  mutate(run3 = run3 - 3*a) %>%
  mutate(run4 = run4 - 4*a) %>%
  mutate(run5 = run5 - 6*a) %>%
  mutate(run6 = run6 - 7*a) %>%
  mutate(temperature = temperature + 3*a) %>%
  mutate(temperature_input = temperature_input + 7*a)
colnames(fish2_state2)[2:9] <- legend_label
fish2_state2 <- fish2_state2 %>% pivot_longer(cols = -time_id) %>% arrange(time_id, name)
Fig_FishState2 <- fish2_state2 %>%
  ggplot(aes(x = time_id, y = value, color = name)) + 
  geom_hline(yintercept = c(0,-1,-3,-4,-6,-7)*a, linetype = 2) +
  geom_line() +
  theme(axis.text.y = element_blank()) +
  scale_color_manual(values = c("darkred", "darkred",
                                "orange2", "orange2",
                                "darkblue", "darkblue", "red4", "red3")) +
  xlab("Time step") +
  NULL


# Reservoir state all
state_legend <- get_legend(Fig_LorenzState0)
Fig_ReservoirState <- plot_grid(Fig_LorenzState0 + theme(legend.position = "none"),
                                Fig_FishState1 + theme(legend.position = "none"),
                                Fig_FishState2 + theme(legend.position = "none"),
                                plot_grid(NULL, state_legend, ncol = 2, rel_widths = c(1,2)),
                                rel_widths = c(1, 1),
                                ncol = 2, labels = c("a","b","c", NULL))

# Figure: Prediction capacity
fs <- 12
Fig_PredictTP <- plot_grid(lorenz_fig[[1]] + theme(legend.position = c(0.1, 0.9), plot.title = element_text(size = fs)) +
                           ylim(0.75,1.3) +
                             ggtitle(expression(paste("Lorenz attractor"))) +
                             xlab("Prediction (min)"),
                           fish1_fig[[1]] + ylim(0.9, 1.8) +
                             ggtitle(expression(paste("Flatfish: ", italic("Paralichthys olivaceus")))) +
                             theme(legend.position = "none", plot.title = element_text(size = fs)),
                           fish2_fig[[1]] + ylim(0.45,1.3) +
                             ggtitle(expression(paste("Jack mackerel : ", italic("Trachrus japonicus")))) +
                             theme(legend.position = "none", plot.title = element_text(size = fs)),
                           ncol = 3, align = "hv", labels = c("a", "b", "c"))
Fig_PredictTS <- plot_grid(lorenz_fig[[4]] + xlab("Time step") +
                             ggtitle(expression("Prediction of 22 time step future (Lorenz)")),
                           fish1_fig[[4]] + xlab("Time step") +
                             ggtitle(expression("Prediction of 95 weeks future (flatfish)")),
                           fish2_fig[[4]] + xlab("Time step") +
                             ggtitle(expression("Prediction of 20 weeks future (Japanese jack mackerel)")),
                           ncol = 1, align = "hv", labels = c("d", "e", "f"))
Fig_Predict <- plot_grid(Fig_PredictTP, Fig_PredictTS,
                         rel_heights = c(1,3),
                         ncol = 1)




# <----------------------------------------------------> #
#                  Save figures
# <----------------------------------------------------> #
ggsave(file = sprintf("%s/Fig_GrowthTemp.pdf", fig_output_dir),
       device = cairo_pdf,
       plot = Fig_TempEffect, width = 8, height = 9)
ggsave(file = sprintf("%s/Fig_GrowthMedium.pdf", fig_output_dir),
       device = cairo_pdf,
       plot = Fig_MediumEffect, width = 8, height = 9)
ggsave2(sprintf("%s/Fig_InRealTime_CellCount.pdf", fig_output_dir),
        Fig_CellCount, width = 8, height = 10)
ggsave2(sprintf("%s/Fig_InRealTime_ReservoirState.pdf", fig_output_dir),
        Fig_ReservoirState, width = 12, height = 12)
ggsave2(sprintf("%s/Fig_InRealTime_Predict2.pdf", fig_output_dir),
        Fig_Predict, width = 10, height = 12)

