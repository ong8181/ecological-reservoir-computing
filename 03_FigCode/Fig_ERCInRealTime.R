####
#### Assemble figures for ERC paper
#### Real-time ERC figures
####

# Load library
library(tidyverse); packageVersion("tidyverse") # 1.3.1, 2021.8.5
library(cowplot); packageVersion("cowplot") # 1.1.1, 2021.8.5
library(ggsci); packageVersion("ggsci") # 2.9, 2021.3.30
library(viridis); packageVersion("viridis") # 0.6.1, 2021.8.5
library(magick); packageVersion("magick") # 2.7.2, 2021.8.5
options('tibble.print_max' = 20)
theme_set(theme_cowplot())


# <----------------------------------------------------> #
#            Specify input/output directory
# <----------------------------------------------------> #
# Create figure output directory
fig_output_dir <- "0_FormattedFigs"
temp_input_dir1 <- "../02_ERCrealtime/v202103_runif_all/00_RunData/v20210226_runif01"
fig_input_dir1 <- "../02_ERCrealtime/v202103_runif_all/03_MeasureMCOut"
fig_input_dir2 <- "../02_ERCrealtime/v202103_lorenz_all/03_PredictionOut"
fig_input_dir3 <- "../02_ERCrealtime/v202103_fishts_all/03_PredictionOut"
fig_input_dir4 <- "../02_ERCrealtime/v202103_runif_all/02_MeasureMCOut"
fig_input_dir5 <- "../02_ERCrealtime/v202103_lorenz_all/02_PredictionOut"
fig_input_dir6 <- "../02_ERCrealtime/v202103_fishts_all/02_PredictionOut"
fig_input_dir7 <- "../02_ERCrealtime/v202105_fishts_all/03_PredictionOut"
fig_input_dir8 <- "../02_ERCrealtime/v202103_runif_all/04_ThermometerOut"
fig_input_dir9 <- "../02_ERCrealtime/v202103_runif_all/04_ThermometerOut"
dir.create(fig_output_dir)


# <----------------------------------------------------> #
#                   Load figure data
# <----------------------------------------------------> #
# Experimental setup illustration
fig_setup <- image_read("0_Illustrations/ERCpaper_Fig4_v7.jpg") 

# Random uniform input
runif_fig <- readRDS(sprintf("%s/FigObject.obj", fig_input_dir1))
runif_parm <- readRDS(sprintf("%s/ParameterList.obj", fig_input_dir1))
runif_result <- readRDS(sprintf("%s/ReservoirResult.obj", fig_input_dir1))
runif_state <- readRDS(sprintf("%s/ReservoirState.obj", fig_input_dir1))
runif_raw <- readRDS(sprintf("%s/ReservoirState0.obj", fig_input_dir4))
runif_example <- readRDS(sprintf("%s/ResultExample.obj", fig_input_dir1))
temp_input_df <- read_csv(sprintf("%s/03_CompileDataOut/data_all.csv", temp_input_dir1))[1:(5*256),]
# Tetrahymena thermometer
thermo_raw_fig <- readRDS(sprintf("%s/FigObject.obj", fig_input_dir8))
thermo_5min_fig <- readRDS(sprintf("%s/FigObject.obj", fig_input_dir9))

# Lorenz attractor
lorenz_fig <- readRDS(sprintf("%s/FigObject.obj", fig_input_dir2))
lorenz_parm <- readRDS(sprintf("%s/ParameterList.obj", fig_input_dir2))
lorenz_result <- readRDS(sprintf("%s/ReservoirResult.obj", fig_input_dir2))
lorenz_state <- readRDS(sprintf("%s/ReservoirState.obj", fig_input_dir2))
lorenz_example <- readRDS(sprintf("%s/ResultExample.obj", fig_input_dir2))

# Fish catch time-series (Hirame)
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

# <----------------------------------------------------> #
#                  Format figures
# <----------------------------------------------------> #
# Experimental setup
Fig_ExpImage <- ggdraw() + draw_image(fig_setup)

## Add value range for temperature
m_input_t <- mean(temp_input_df$temperature_input)
sd_input_t <- sd(temp_input_df$temperature_input)
m_med_t <- mean(temp_input_df$temperature)
sd_med_t <- sd(temp_input_df$temperature)

#runif_raw$dif_run1_2 <- abs(runif_raw$run1 - runif_raw$run2)
# Figure: Echo State Property
a <- 1.2
a2 <- 1.3
a3 <- 0.8
runif_state2 <- runif_raw %>% 
  mutate(run2 = run2 - 1*a) %>%
  mutate(run3 = run3*2 - 3*a) %>%
  mutate(run4 = run4*2 - 4*a) %>%
  mutate(run5 = run5*2 - 6*a) %>%
  mutate(run6 = run6*2 - 7*a) %>%
  mutate(run11 = run11*3 - 9*a) %>%
  mutate(temperature = temperature/a2 + 3*a) %>%
  mutate(temperature_input = temperature_input/a3 + 8*a)
legend_label <- c("Temperature (input)","Temperature (medium)",
                  "Med. nutrient 1", "Med. nutrient 2",
                  "Low nutrient 1", "Low nutrient 2",
                  "High nutrient 1", "High nutrient 2", # "High nutrient 3",
                  "Response to different inputs")
colnames(runif_state2)[2:10] <- legend_label
runif_state2 <- runif_state2 %>% pivot_longer(cols = -time_id)
runif_state2$name <- factor(runif_state2$name, levels = legend_label)

## Calculate y-axis values for temperature
min_med_t <- (14 - m_med_t)/sd_med_t/a2 + 3*a
max_med_t <- (25 - m_med_t)/sd_med_t/a2 + 3*a
min_input_t <- (10 - m_input_t)/sd_input_t/a3 + 8*a
max_input_t <- (25 - m_input_t)/sd_input_t/a3 + 8*a

## Generate figure
Fig_RunifState0 <- runif_state2 %>%
  ggplot(aes(x = time_id, y = value, color = name)) + 
  geom_hline(yintercept = c(0, -1, -3, -4, -6, -7, -9)*a, linetype = 2) +
  geom_line(alpha = 1) +
  geom_segment(x = -35, xend = -35, y = min_med_t, yend = max_med_t, alpha = 0.2, linewidth = 0.3, color = "gray10") +
  geom_segment(x = -45, xend = -25, y = min_med_t, yend = min_med_t, alpha = 0.2, linewidth = 0.3, color = "gray10") +
  geom_segment(x = -45, xend = -25, y = max_med_t, yend = max_med_t, alpha = 0.2, linewidth = 0.3, color = "gray10") +
  annotate("text", x = -90, y = min_med_t, label = expression(paste("14", degree, "C"))) +
  annotate("text", x = -90, y = max_med_t, label = expression(paste("25", degree, "C"))) +
  geom_segment(x = -35, xend = -35, y = min_input_t, yend = max_input_t, alpha = 0.2, linewidth = 0.3, color = "gray10") +
  geom_segment(x = -45, xend = -25, y = min_input_t, yend = min_input_t, alpha = 0.2, linewidth = 0.3, color = "gray10") +
  geom_segment(x = -45, xend = -25, y = max_input_t, yend = max_input_t, alpha = 0.2, linewidth = 0.3, color = "gray10") +
  annotate("text", x = -90, y = min_input_t, label = expression(paste("10", degree, "C"))) +
  annotate("text", x = -90, y = max_input_t, label = expression(paste("25", degree, "C"))) +
  xlim(-90, 1300) +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = "top") +
  #scale_color_manual(values = c("red4", "red3",
  #                              "royalblue", "red3",
  #                              "royalblue", "red3",
  #                              "royalblue", "red3",
  #                              "gray20")) +
  #annotate("text", x = -30, y = 1.5, label = "4% of modified Neff", hjust = 0, size = 6) + 
  #annotate("text", x = -30, y = -1.25, label = "1.6% of modified Neff", hjust = 0, size = 6) + 
  #annotate("text", x = -30, y = -4, label = "10% of modified Neff", hjust = 0, size = 6) + 
  scale_color_manual(values = c("red4", "red3",
                                "darkblue", "darkblue",
                                "orange2", "orange2",
                                "darkred", "darkred", #"darkred",
                                "gray20"), name = NULL) +
  xlab("Time step") +
  NULL

# ESP: Time series plot
Fig_ESP1 <- ggplot(runif_raw, aes(x = time_id, y = abs(run1 - run2))) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "gam", formula = y ~ s(x, k=3), color = "red3", se = FALSE) + 
  xlab("Time step") + ylab("Abs(Run1 - Run2)") +
  ylim(0, 1.1) +
  NULL
Fig_ESP2 <- ggplot(runif_raw, aes(x = time_id, y = abs(run3 - run4))) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "gam", formula = y ~ s(x, k=3), color = "red3", se = FALSE) + 
  xlab("Time step") + ylab("Abs(Run3 - Run4)") +
  ylim(0, 1.1) +
  NULL
Fig_ESP3 <- ggplot(runif_raw, aes(x = time_id, y = abs(run5 - run6))) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "gam", formula = y ~ s(x, k=3), color = "red3", se = FALSE) + 
  xlab("Time step") + ylab("Abs(Run5 - Run6)") +
  ylim(0, 1.1) +
  NULL
Fig_ESP4 <- ggplot(runif_raw, aes(x = time_id, y = abs(run1 - run11))) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "gam", formula = y ~ s(x, k=3), color = "red3", se = FALSE) + 
  xlab("Time step") + ylab("Abs(Run1 - Run11)") +
  ylim(0, 1.1) +
  NULL

# Scatter plot
runif_sub <- runif_raw[780:nrow(runif_raw),]
Fig_ESP5 <- ggplot(runif_sub, aes(x = run1, y = run2)) +
  geom_point(alpha = 0.3) +
  geom_abline(intercept = 0, slope = 1, linetype = 2) + 
  xlab("States of Run1") + ylab("States of Run2") +
  NULL
Fig_ESP6 <- ggplot(runif_sub, aes(x = run3, y = run4)) +
  geom_point(alpha = 0.3) +
  geom_abline(intercept = 0, slope = 1, linetype = 2) + 
  xlab("States of Run3") + ylab("States of Run4") +
  NULL
Fig_ESP7 <- ggplot(runif_sub, aes(x = run5, y = run6)) +
  geom_point(alpha = 0.3) +
  geom_abline(intercept = 0, slope = 1, linetype = 2) + 
  xlab("States of Run5") + ylab("States of Run6") +
  NULL
Fig_ESP8 <- ggplot(runif_sub, aes(x = run1, y = run11)) +
  geom_point(alpha = 0.3) +
  geom_abline(intercept = 0, slope = 1, linetype = 2) + 
  xlab("States of Run1") + ylab("States of Run11") +
  NULL

Fig_ESPs <- plot_grid(Fig_ESP1, Fig_ESP5,
                      Fig_ESP2, Fig_ESP6,
                      Fig_ESP3, Fig_ESP7,
                      Fig_ESP4, Fig_ESP8,
                      ncol = 2, labels = c("auto"),
                      rel_widths = c(2,1))


# <----------------------------------------------------> #
#                    Align panels
# <----------------------------------------------------> #
# Figure: Experimental setup + reservoir state
Fig_ExpSetup <- plot_grid(Fig_ExpImage,
                          Fig_RunifState0,
                          labels = c(NA, "f"),
                          ncol = 1, rel_heights = c(1.5,1))


# Figure: Memory capacity
Fig_Memory1 <- plot_grid(runif_fig[[1]] +
                           xlim(0,75) +
                           ylab(expression(paste("Coef. of det. (", R^{2}, ")"))) +
                           theme(legend.position = c(0.2, 0.8)) +
                           ggtitle(NULL),
                         runif_fig[[2]], align = "hv",
                         ncol=2, labels = c("a","b"))
Fig_Thermo <- plot_grid(thermo_raw_fig[[4]] +
                          ggtitle(expression(paste(italic("Tetrahymena"), " thermometer measuring the medium temp."))) +
                          xlab("Time step") +
                          theme(legend.position = "top", plot.title = element_text(size = 9, face = "plain")),
                        thermo_raw_fig[[2]] +
                          ylab("Predicted temperature\n(normalized)") +
                          xlab("Observed temperature\n(normalized)"),
                        align = "hv", axis = "lrtb", rel_widths = c(1.8,1),
                        ncol=2, labels = c("d","e"))
Fig_MemoryCapacity <- plot_grid(Fig_Memory1,
                                runif_fig[[4]] +
                                  ggtitle("Remembering temperature 5 minutes ago") +
                                  xlab("Time step") +
                                  theme(legend.position = "top", plot.title = element_text(size = 9, face = "plain")),
                                Fig_Thermo,
                                ncol = 1,
                                #align = "hv", axis = "lrtb",
                                labels = c(NA, "C", NA))
fs <- 9
Fig_PredictTP <- plot_grid(lorenz_fig[[1]] + theme(legend.position = c(0.1, 0.9), plot.title = element_text(size = fs)) + coord_cartesian(ylim = c(0.75,1.5)) + xlim(0, 25) +
                             ggtitle(expression(paste("Lorenz attractor"))) + xlab("Prediction (time steps)"),
                           fish1_fig[[1]] + coord_cartesian(ylim = c(0.4,1.65)) +
                             ggtitle(expression(paste("Flatfish: ", italic("Paralichthys olivaceus")))) +
                             theme(legend.position = "none", plot.title = element_text(size = fs)),
                           fish2_fig[[1]] + coord_cartesian(ylim = c(0.5,1.2)) +
                             ggtitle(expression(paste("Jack mackerel : ", italic("Trachrus japonicus")))) +
                             theme(legend.position = "none", plot.title = element_text(size = fs)),
                           ncol = 3, align = "hv", labels = c("a", "b", "c"))
Fig_PredictTS <- plot_grid(lorenz_fig[[4]] + xlab("Time step") +
                             ggtitle(expression("Prediction of 15 time step future (Lorenz)")),
                           fish1_fig[[4]] + xlab("Time step") +
                             ggtitle(expression("Prediction of 19 weeks future (flatfish)")),
                           fish2_fig[[4]] + xlab("Time step") +
                             ggtitle(expression("Prediction of 30 weeks future (Japanese jack mackerel)")),
                           ncol = 1, align = "hv", labels = c("d", "e", "f"))
Fig_Predict <- plot_grid(Fig_PredictTP, Fig_PredictTS,
                         rel_heights = c(1,3),
                         ncol = 1)

# Make Large figure
Fig_MC_Predict <- plot_grid(Fig_MemoryCapacity, NULL, Fig_Predict,
                            ncol = 3, rel_widths = c(1,0.1,1))

# <----------------------------------------------------> #
#                  Save figures
# <----------------------------------------------------> #
ggsave2(sprintf("%s/Fig_InRealTime_Setup.pdf", fig_output_dir),
        Fig_ExpSetup, width = 12, height = 14)
ggsave2(sprintf("%s/Fig_InRealTime_StateESP.pdf", fig_output_dir),
        Fig_ESPs, width = 9, height = 11)
ggsave2(sprintf("%s/Fig_InRealTime_MC.pdf", fig_output_dir),
        Fig_MemoryCapacity, width = 10, height = 12)
ggsave2(sprintf("%s/Fig_InRealTime_Predict.pdf", fig_output_dir),
        Fig_Predict, width = 10, height = 12)
ggsave2(sprintf("%s/Fig_InRealTime_MCPredict.pdf", fig_output_dir),
        Fig_MC_Predict, width = 17, height = 12)
