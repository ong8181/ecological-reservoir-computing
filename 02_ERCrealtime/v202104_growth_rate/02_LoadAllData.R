####
#### Real-time ERC: No.2 Load all data
#### For different medium conditions
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
d01 <- read_csv("00_RunData/v20210413_const25C_m1/03_CompileDataOut/data_all.csv")
d02 <- read_csv("00_RunData/v20210415_const25C_m2/03_CompileDataOut/data_all.csv")
d03 <- read_csv("00_RunData/v20210416_const25C_m3/03_CompileDataOut/data_all.csv")
d04 <- read_csv("00_RunData/v20210417_const25C_m4/03_CompileDataOut/data_all.csv")
d05 <- read_csv("00_RunData/v20210418_const25C_m5/03_CompileDataOut/data_all.csv")
d06 <- read_csv("00_RunData/v20210513_const25C_m6/03_CompileDataOut/data_all.csv")


# <---------------------------------------------> #
#             Estimate growth rate
# <---------------------------------------------> #
## Define time window for growth rate estimation
# Approximation: x(t) = x0 * exp(k*t)
# Nonlinear regression
## Modified Neff * 1, Temperature = 25C const.
plot(d01$density, type = "l")
TW1 = 101:700
gr01 <- nls(d01$density[TW1] ~ a * exp(k*(1:length(TW1))),
            start = c(a = 1, k = 0.001), trace = F)
k01 <- summary(gr01)$coefficients["k","Estimate"]
## Visualize predictions
gr01_df <- data.frame(time = 1:length(TW1),
                      obs = d01$density[TW1],
                      pred = predict(gr01))

g1 <- ggplot(gr01_df, aes(x = time, y = obs)) + geom_point(alpha = 0.5) +
  geom_line(data = gr01_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,250) +
  ggtitle(expression(paste("Modified Neff 100% at 25", degree, "C"))) + 
  annotate("text", x = 100, y = 200, label = sprintf("k = %0.5f", k01)) +
  annotate("text", x = 100, y = 170, label = sprintf("Doubling time = %0.1f min", log(2)/k01)) +
  NULL

## Modified Neff * 0.04, Temperature = 20C const.
plot(d02$density, type = "l")
TW2 = 201:800
gr02 <- nls(d02$density[TW2] ~ a * exp(k*(1:length(TW2))),
            start = c(a = 1, k = 0.001), trace = F)
k02 <- summary(gr02)$coefficients["k","Estimate"]; log(2)/k02
## Visualize predictions
gr02_df <- data.frame(time = 1:length(TW2),
                      obs = d02$density[TW2],
                      pred = predict(gr02))
g2 <- ggplot(gr02_df, aes(x = time, y = obs + 0.5)) + geom_point(alpha = 0.5) +
  geom_line(data = gr02_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,250) +
  ggtitle(expression(paste("Modified Neff 4% at 25", degree, "C"))) + 
  annotate("text", x = 100, y = 200, label = sprintf("k = %0.5f", k02)) +
  annotate("text", x = 100, y = 170, label = sprintf("Doubling time = %0.1f min", log(2)/k02)) +
  NULL

## Modified Neff * 0.016, Temperature = 10C const.
plot(d03$density, type = "l")
TW3 = 301:900
gr03 <- nls(d03$density[TW3] ~ a * exp(k*(1:length(TW3))),
            start = c(a = 1, k = 0.001), trace = F)
k03 <- summary(gr03)$coefficients["k","Estimate"]
## Visualize predictions
gr03_df <- data.frame(time = 1:length(TW3),
                      obs = d03$density[TW3],
                      pred = predict(gr03))
g3 <- ggplot(gr03_df, aes(x = time, y = obs + 0.5)) + geom_point(alpha = 0.5) +
  geom_line(data = gr03_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,250) +
  ggtitle(expression(paste("Modified Neff 1.6% at 25", degree, "C"))) + 
  annotate("text", x = 100, y = 200, label = sprintf("k = %0.5f", k03)) +
  annotate("text", x = 100, y = 170, label = sprintf("Doubling time = %0.1f min", log(2)/k03)) +
  NULL

## Modified Neff * 0.05, Temperature = 25C const.
plot(d04$density, type = "l")
TW4 = 251:850
gr04 <- nls(d04$density[TW4] ~ a * exp(k*(1:length(TW4))),
            start = c(a = 1, k = 0.001), trace = F)
k04 <- summary(gr04)$coefficients["k","Estimate"]
## Visualize predictions
gr04_df <- data.frame(time = 1:length(TW4),
                      obs = d04$density[TW4],
                      pred = predict(gr04))
g4 <- ggplot(gr04_df, aes(x = time, y = obs + 0.5)) + geom_point(alpha = 0.5) +
  geom_line(data = gr04_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,250) +
  ggtitle(expression(paste("Modified Neff 5% at 25", degree, "C"))) + 
  annotate("text", x = 100, y = 200, label = sprintf("k = %0.5f", k04)) +
  annotate("text", x = 100, y = 170, label = sprintf("Doubling time = %0.1f min", log(2)/k04)) +
  NULL

## Modified Neff * 0.3, Temperature = 25C const.
plot(d05$density, type = "l")
TW5 = 101:700
gr05 <- nls(d05$density[TW5] ~ a * exp(k*(1:length(TW5))),
            start = c(a = 1, k = 0.001), trace = F)
k05 <- summary(gr05)$coefficients["k","Estimate"]
## Visualize predictions
gr05_df <- data.frame(time = 1:length(TW5),
                      obs = d05$density[TW5],
                      pred = predict(gr05))
g5 <- ggplot(gr05_df, aes(x = time, y = obs + 0.5)) + geom_point(alpha = 0.5) +
  geom_line(data = gr05_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,250) +
  ggtitle(expression(paste("Modified Neff 30% at 25", degree, "C"))) + 
  annotate("text", x = 100, y = 200, label = sprintf("k = %0.5f", k05)) +
  annotate("text", x = 100, y = 170, label = sprintf("Doubling time = %0.1f min", log(2)/k05)) +
  NULL

## Modified Neff * 0.1, Temperature = 25C const.
plot(d06$density, type = "l")
TW6 = 151:750
gr06 <- nls(d06$density[TW6] ~ a * exp(k*(1:length(TW6))),
            start = c(a = 1, k = 0.001), trace = F)
k06 <- summary(gr06)$coefficients["k","Estimate"]
## Visualize predictions
gr06_df <- data.frame(time = 1:length(TW6),
                      obs = d06$density[TW6],
                      pred = predict(gr06))
g6 <- ggplot(gr06_df, aes(x = time, y = obs + 0.5)) + geom_point(alpha = 0.5) +
  geom_line(data = gr06_df, aes(x = time, y = pred), color = "red3") +
  xlab("Time (min)") + ylab("Cell density (/image)") + ylim(0,250) +
  ggtitle(expression(paste("Modified Neff 10% at 25", degree, "C"))) + 
  annotate("text", x = 100, y = 200, label = sprintf("k = %0.5f", k06)) +
  annotate("text", x = 100, y = 170, label = sprintf("Doubling time = %0.1f min", log(2)/k06)) +
  NULL


## Standardize all density
gr01_df <- gr01_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])
gr02_df <- gr02_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])
gr03_df <- gr03_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])
gr04_df <- gr04_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])
gr05_df <- gr05_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])
gr06_df <- gr06_df %>% mutate(obs_rel = obs/pred[1], pred_rel = pred/pred[1])


# <---------------------------------------------> #
#              Visualize patterns
# <---------------------------------------------> #
## Visualize growth patterns
colors <- c("×1.000" = "red",
            "×0.300" = "red3",
            "×0.100" = "darkred",
            "×0.050" = "orange",
            "×0.040" = "chartreuse3",
            "×0.016" = "royalblue",
            NULL)

g_ts <- ggplot() +
  geom_point(data = gr01_df, aes(x = time, y = obs_rel, color = "×1.000"), alpha = 0.5) +
  geom_point(data = gr02_df, aes(x = time, y = obs_rel, color = "×0.040"), alpha = 0.5) +
  geom_point(data = gr03_df, aes(x = time, y = obs_rel, color = "×0.016"), alpha = 0.5) +
  geom_point(data = gr04_df, aes(x = time, y = obs_rel, color = "×0.050"), alpha = 0.5) +
  geom_point(data = gr05_df, aes(x = time, y = obs_rel, color = "×0.300"), alpha = 0.5) +
  geom_point(data = gr06_df, aes(x = time, y = obs_rel, color = "×0.100"), alpha = 0.5) +
  geom_line(data = gr01_df, aes(x = time, y = pred_rel), color = "gray20") +
  geom_line(data = gr02_df, aes(x = time, y = pred_rel), color = "gray20") +
  geom_line(data = gr03_df, aes(x = time, y = pred_rel), color = "gray20") +
  geom_line(data = gr04_df, aes(x = time, y = pred_rel), color = "gray20") +
  geom_line(data = gr05_df, aes(x = time, y = pred_rel), color = "gray20") +
  geom_line(data = gr06_df, aes(x = time, y = pred_rel), color = "gray20") +
  xlab("Time (min)") + ylab("Relative density") + ylim(0,23) +
  labs(color = "Medium\nconcentration") +
  scale_color_manual(values = colors) +
  theme(legend.position = c(0.05, 0.8)) +
  annotate("text", x = 200, y = 17.8,
  label = sprintf("Td (×0.016) = %0.0f min\nTd (×0.040) = %0.0f min\nTd (×0.050) = %0.0f min\nTd (×0.100) = %0.0f min\nTd (×0.300) = %0.0f min\nTd (×1.000) = %0.0f min",
                 log(2)/k03, log(2)/k02, log(2)/k04, log(2)/k06, log(2)/k05, log(2)/k01)) +
  #scale_y_log10() +
  NULL

## Growth rates
gr_table <- as.data.frame(rbind(summary(gr01)$coefficients["k",],
                                summary(gr05)$coefficients["k",],
                                summary(gr06)$coefficients["k",],
                                summary(gr04)$coefficients["k",],
                                summary(gr02)$coefficients["k",],
                                summary(gr03)$coefficients["k",],
                                NULL))
colnames(gr_table) <- c("k", "std", "t", "p_val")
gr_table$medium_conc <- c(100, 30, 10, 5, 4, 1.6)
gr_table$td <- log(2)/gr_table$k
g_gr <- ggplot(gr_table, aes(x = medium_conc, y = k)) +
  geom_point() +
  geom_errorbar(aes(ymin = k - std*1.96, ymax = k + std*1.96), width = .02) +
  xlab("Medium concentration (%)") +
  ylab("Growth constant (k)") +
  scale_x_log10() +
  annotate("text", x = 5, y = 0.0055, label = "X[t] == X[0] * e ^ {kt}", parse = T, size = 6) +
  NULL
g_td <- ggplot(gr_table, aes(x = medium_conc, y = td)) +
  geom_point() +
  geom_errorbar(aes(ymin = log(2)/(k - std*1.96), ymax = log(2)/(k + std*1.96)), width = .02) +
  xlab("Medium concentration (%)") +
  ylab("Doubling time (min)") +
  scale_x_log10() +
  #scale_y_log10(limits = c(100,550)) +
  NULL
g_parm <- plot_grid(g_gr, g_td, ncol = 2, align = "hv")
g_all <- plot_grid(g_ts, g_parm, ncol = 1, rel_heights = c(1,0.8))

# <---------------------------------------------> #
#                 Save output
# <---------------------------------------------> #
ggsave(file = sprintf("%s/GrowthRateMedium_ts.pdf", output_folder),
       device = cairo_pdf,
       plot = g_ts, width = 8, height = 5)
ggsave(file = sprintf("%s/GrowthRateMedium_parms.pdf", output_folder),
       device = cairo_pdf,
       plot = g_parm, width = 8, height = 5)
ggsave(file = sprintf("%s/GrowthRateMedium_all.pdf", output_folder),
       device = cairo_pdf,
       plot = g_all, width = 8, height = 9)

# Save objects
g_list <- list(g_ts, g_gr, g_td)
saveRDS(g_list, file = sprintf("%s/MediumDependence.obj", output_folder))

# Save workspace and objects
save.image(sprintf("%s/%s.RData", output_folder, output_folder))

# Save session info
writeLines(capture.output(sessionInfo()),
           sprintf("00_SessionInfo/%s_SessionInfo_%s.txt", output_folder, output_folder, substr(Sys.time(), 1, 10)))
 
