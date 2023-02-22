####
#### Real-time ERC: No.1 Load all data
####

# Load library
library(tidyverse); packageVersion("tidyverse") # 1.3.0, 2020.11.17
library(lubridate); packageVersion("lubridate") # 1.7.9, 2020.11.17
library(cowplot); packageVersion("cowplot") # 1.1.0, 2020.11.17
library(ggsci); packageVersion("ggsci") # 2.9, 2020.12.25
library(viridis); packageVersion("viridis") # 0.5.1, 2021.3.9
theme_set(theme_cowplot())

# Generate output folder
od <- basename(rstudioapi::getSourceEditorContext()$path)
(output_folder <- paste0(str_sub(od, end = -3), "Out")); rm(od)
dir.create(output_folder)
dir.create("00_SessionInfo")


# <---------------------------------------------> #
#                   Load data 
# <---------------------------------------------> #
## Random input set1
d01 <- read_csv("00_RunData/v20210402_const30C_m1/03_CompileDataOut/data_all.csv")
d02 <- read_csv("00_RunData/v20210409_const20C_m1/03_CompileDataOut/data_all.csv")
d03 <- read_csv("00_RunData/v20210410_const10C_m1/03_CompileDataOut/data_all.csv")
d04 <- read_csv("00_RunData/v20210413_const25C_m1/03_CompileDataOut/data_all.csv")
d05 <- read_csv("00_RunData/v20210414_const15C_m1/03_CompileDataOut/data_all.csv")


# <---------------------------------------------> #
#             Estimate growth rate
# <---------------------------------------------> #
## Define time window for growth rate estimation
# Approximation: x(t) = x0 * exp(k*t)
# Nonlinear regression
## Modified Neff * 1, Temperature = 30C const.
plot(d01$density, type = "l")
TW1 = 201:800
gr01 <- nls(d01$density[TW1] ~ a * exp(k*(1:length(TW1))),
            start = c(a = 1, k = 0.001), trace = F)
k01 <- summary(gr01)$coefficients["k","Estimate"]
## Visualize predictions
gr01_df <- data.frame(time = 1:length(TW1),
                      obs = d01$density[TW1],
                      pred = predict(gr01))
g1 <- ggplot(gr01_df, aes(x = time, y = obs)) + geom_point(alpha = 0.5) +
  geom_line(data = gr01_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,530) +
  ggtitle(expression(paste("Modified Neff at 30", degree, "C"))) + 
  annotate("text", x = 100, y = 400, label = sprintf("k = %0.5f", k01)) +
  annotate("text", x = 100, y = 360, label = sprintf("Doubling time = %0.1f min", log(2)/k01)) +
  NULL
  
## Modified Neff * 1, Temperature = 20C const.
plot(d02$density, type = "l")
TW2 = 201:800
gr02 <- nls(d02$density[TW2] ~ a * exp(k*(1:length(TW2))),
            start = c(a = 1, k = 0.001), trace = F)
k02 <- summary(gr02)$coefficients["k","Estimate"]
## Visualize predictions
gr02_df <- data.frame(time = 1:length(TW2),
                      obs = d02$density[TW2],
                      pred = predict(gr02))
g2 <- ggplot(gr02_df, aes(x = time, y = obs + 0.5)) + geom_point(alpha = 0.5) +
  geom_line(data = gr02_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,530) +
  ggtitle(expression(paste("Modified Neff at 20", degree, "C"))) + 
  annotate("text", x = 100, y = 400, label = sprintf("k = %0.5f", k02)) +
  annotate("text", x = 100, y = 360, label = sprintf("Doubling time = %0.1f min", log(2)/k02)) +
  NULL

## Modified Neff * 1, Temperature = 10C const.
plot(d03$density, type = "l")
TW3 = 701:1300
gr03 <- nls(d03$density[TW3] ~ a * exp(k*(1:length(TW3))),
            start = c(a = 1, k = 0.001), trace = F)
k03 <- summary(gr03)$coefficients["k","Estimate"]
## Visualize predictions
gr03_df <- data.frame(time = 1:length(TW3),
                      obs = d03$density[TW3],
                      pred = predict(gr03))
g3 <- ggplot(gr03_df, aes(x = time, y = obs + 0.5)) + geom_point(alpha = 0.5) +
  geom_line(data = gr03_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,530) +
  ggtitle(expression(paste("Modified Neff at 11", degree, "C"))) + 
  annotate("text", x = 100, y = 400, label = sprintf("k = %0.5f", k03)) +
  annotate("text", x = 100, y = 360, label = sprintf("Doubling time = %0.1f min", log(2)/k03)) +
  NULL

## Modified Neff * 1, Temperature = 25C const.
plot(d04$density, type = "l")
TW4 = 101:700
gr04 <- nls(d04$density[TW4] ~ a * exp(k*(1:length(TW4))),
            start = c(a = 1, k = 0.001), trace = F)
k04 <- summary(gr04)$coefficients["k","Estimate"]
## Visualize predictions
gr04_df <- data.frame(time = 1:length(TW4),
                      obs = d04$density[TW4],
                      pred = predict(gr04))
g4 <- ggplot(gr04_df, aes(x = time, y = obs + 0.5)) + geom_point(alpha = 0.5) +
  geom_line(data = gr04_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,530) +
  ggtitle(expression(paste("Modified Neff at 25", degree, "C"))) + 
  annotate("text", x = 100, y = 400, label = sprintf("k = %0.5f", k04)) +
  annotate("text", x = 100, y = 360, label = sprintf("Doubling time = %0.1f min", log(2)/k04)) +
  NULL

## Modified Neff * 1, Temperature = 15C const.
plot(d05$density, type = "l")
TW5 = 601:1200
gr05 <- nls(d05$density[TW5] ~ a * exp(k*(1:length(TW5))),
            start = c(a = 1, k = 0.001), trace = F)
k05 <- summary(gr05)$coefficients["k","Estimate"]
## Visualize predictions
gr05_df <- data.frame(time = 1:length(TW5),
                      obs = d05$density[TW5],
                      pred = predict(gr05))
g5 <- ggplot(gr05_df, aes(x = time, y = obs + 0.5)) + geom_point(alpha = 0.5) +
  geom_line(data = gr05_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,530) +
  ggtitle(expression(paste("Modified Neff at 15", degree, "C"))) + 
  annotate("text", x = 100, y = 400, label = sprintf("k = %0.5f", k05)) +
  annotate("text", x = 100, y = 360, label = sprintf("Doubling time = %0.1f min", log(2)/k05)) +
  NULL

## Standardize all density
gr01_df <- gr01_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])
gr02_df <- gr02_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])
gr03_df <- gr03_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])
gr04_df <- gr04_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])
gr05_df <- gr05_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])


# <---------------------------------------------> #
#              Visualize patterns
# <---------------------------------------------> #
## Visualize growth patterns
colors <- c("30˚C" = "red3",
            "25˚C" = "darkred",
            "20˚C" = "orange",
            "15˚C" = "chartreuse3",
            "11˚C" = "royalblue")

g_ts <- ggplot() +
  geom_point(data = gr01_df, aes(x = time, y = obs_rel, color = "30˚C"), alpha = 0.5) +
  geom_point(data = gr04_df, aes(x = time, y = obs_rel, color = "25˚C"), alpha = 0.5) +
  geom_point(data = gr02_df, aes(x = time, y = obs_rel, color = "20˚C"), alpha = 0.5) +
  geom_point(data = gr05_df, aes(x = time, y = obs_rel, color = "15˚C"), alpha = 0.5) +
  geom_point(data = gr03_df, aes(x = time, y = obs_rel, color = "11˚C"), alpha = 0.5) +
  geom_line(data = gr01_df, aes(x = time, y = pred_rel), color = "gray20") +
  geom_line(data = gr04_df, aes(x = time, y = pred_rel), color = "gray20") +
  geom_line(data = gr02_df, aes(x = time, y = pred_rel), color = "gray20") +
  geom_line(data = gr05_df, aes(x = time, y = pred_rel), color = "gray20") +
  geom_line(data = gr03_df, aes(x = time, y = pred_rel), color = "gray20") +
  xlab("Time (min)") + ylab("Relative density") + ylim(0,30) +
  labs(color = "Incubation\ntemperature") +
  scale_color_manual(values = colors) +
  theme(legend.position = c(0.05, 0.8)) +
  annotate("text", x = 150, y = 22.3,
  label = sprintf("Td (11˚C) = %0.0f min\nTd (15˚C) = %0.0f min\nTd (20˚C) = %0.0f min\nTd (25˚C) =  %0.0f min\nTd (30˚C) =  %0.0f min\n",
                 log(2)/k03, log(2)/k05, log(2)/k02, log(2)/k04, log(2)/k01)) +
  #scale_y_log10() +
  NULL

## Growth rates
gr_table <- as.data.frame(rbind(summary(gr01)$coefficients["k",],
                                summary(gr02)$coefficients["k",],
                                summary(gr03)$coefficients["k",],
                                summary(gr04)$coefficients["k",],
                                summary(gr05)$coefficients["k",]))
colnames(gr_table) <- c("k", "std", "t", "p_val")
gr_table$temperature <- c(30, 20, 11, 25, 15)
gr_table$td <- log(2)/gr_table$k
g_gr <- ggplot(gr_table, aes(x = temperature, y = k)) +
  geom_point() +
  geom_errorbar(aes(ymin = k - std*1.96, ymax = k + std*1.96), width = .2) +
  xlab("Medium temperature (˚C)") +
  ylab("Growth constant (k)") +
  annotate("text", x = 15, y = 0.0055, label = "X[t] == X[0] * e ^ {kt}", parse = T, size = 6) +
  NULL
g_td <- ggplot(gr_table, aes(x = temperature, y = td)) +
  geom_point() +
  geom_errorbar(aes(ymin = log(2)/(k - std*1.96), ymax = log(2)/(k + std*1.96)), width = .2) +
  xlab("Medium temperature (˚C)") +
  ylab("Doubling time (min)") +
  #ylim(100,550) +
  scale_y_log10(limits = c(100,550)) +
  NULL
g_parm <- plot_grid(g_gr, g_td, ncol = 2, align = "hv")
g_all <- plot_grid(g_ts, g_parm, ncol = 1, rel_heights = c(1,0.8))


# <---------------------------------------------> #
#                 Save output
# <---------------------------------------------> #
ggsave(file = sprintf("%s/GrowthRateTemp_ts.pdf", output_folder),
       device = cairo_pdf,
       plot = g_ts, width = 8, height = 5)
ggsave(file = sprintf("%s/GrowthRateTemp_parms.pdf", output_folder),
       device = cairo_pdf,
       plot = g_parm, width = 8, height = 5)
ggsave(file = sprintf("%s/GrowthRateTemp_all.pdf", output_folder),
       device = cairo_pdf,
       plot = g_all, width = 8, height = 9)

# Save objects
g_list <- list(g_ts, g_gr, g_td)
saveRDS(g_list, file = sprintf("%s/TempDependence.obj", output_folder))

# Save workspace and objects
save.image(sprintf("%s/%s.RData", output_folder, output_folder))

# Save session info
writeLines(capture.output(sessionInfo()),
           sprintf("00_SessionInfo/%s_SessionInfo_%s.txt", output_folder, output_folder, substr(Sys.time(), 1, 10)))
 
