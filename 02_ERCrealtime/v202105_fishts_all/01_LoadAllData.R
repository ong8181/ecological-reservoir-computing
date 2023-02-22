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
## Lorenz input set1
d01 <- read_csv("00_RunData/v20210519_fish01/03_CompileDataOut/data_all.csv")
d02 <- read_csv("00_RunData/v20210520_fish02/03_CompileDataOut/data_all.csv")
d03 <- read_csv("00_RunData/v20210521_fish03/03_CompileDataOut/data_all.csv")
d04 <- read_csv("00_RunData/v20210522_fish04/03_CompileDataOut/data_all.csv")
d05 <- read_csv("00_RunData/v20210523_fish05/03_CompileDataOut/data_all.csv")
d06 <- read_csv("00_RunData/v20210524_fish06/03_CompileDataOut/data_all.csv")
## Select extracted variable
extracted_var <- "density_rel_resid10"
std_trunclen <- 200

# <---------------------------------------------> #
#             Check and adjust data
# <---------------------------------------------> #
dim(d01); dim(d02); dim(d03); dim(d04)
dim(d05); dim(d06)

# Visualize timing of the first xxx data points
temp_len <- 500
d_temp100 <- tibble(time_id = 1:temp_len,
                    temp1 = d01$temperature[1:temp_len],
                    temp2 = d02$temperature[1:temp_len],
                    temp3 = d03$temperature[1:temp_len],
                    temp4 = d04$temperature[1:temp_len],
                    temp5 = d05$temperature[1:temp_len],
                    temp6 = d06$temperature[1:temp_len],
                    NULL)
g1 <- d_temp100 %>% pivot_longer(cols = -time_id) %>% ggplot(aes(x = time_id, y = value, color = name)) +
  geom_vline(xintercept = seq(1, temp_len, by = 5), color = "gray", alpha = 0.7, size = 0.3) +
  geom_line(alpha = 0.5) +
  geom_point(alpha = 0.5) +
  scale_color_igv() +
  NULL

# Calculate residuals and identify the starting point
get_adjust_id <- function(x, d_len = 500, show_plot = T, lag_range = -10:10){
  adjust_resid <- c()
  for(i in lag_range){
    if(i < 0) adjust_resid <- c(adjust_resid, mean(abs(d01$temperature_input[10:d_len] - dplyr::lead(x$temperature_input[10:d_len], n = -i)), na.rm = T))
    if(i >= 0) adjust_resid <- c(adjust_resid, mean(abs(d01$temperature_input[10:d_len] - dplyr::lag(x$temperature_input[10:d_len], n = i)), na.rm = T))
  }
  if(show_plot) plot(lag_range, adjust_resid, type = "b", xlab = "lag", ylab = "MAE")
  return(c(lag_range)[as.integer(which.min(adjust_resid))])
}
## Adjust ID for input 1
d01_id <- get_adjust_id(d01)
d02_id <- get_adjust_id(d02)
d03_id <- get_adjust_id(d03)
d04_id <- get_adjust_id(d04)
d05_id <- get_adjust_id(d05)
d06_id <- get_adjust_id(d06)


# <---------------------------------------------> #
#             Compile new data.frame
# <---------------------------------------------> #
d_id_all <- c(d01_id, d02_id, d03_id, d04_id, d05_id, d06_id,
              NULL)
d_all00 <- list(d01, d02, d03, d04, d05, d06,
                NULL)
d_all <- tibble(time_id = 1:nrow(d01),
                temperature_input = d01$temperature_input,
                temperature = d01$temperature,
                run1 = unlist(d01[,extracted_var]))
for(i in 2:length(d_id_all)){
  id_i <- d_id_all[i]
  d_i <- d_all00[[i]]
  if(id_i < 0){
    d_all <- d_all %>% mutate(dplyr::lead(d_i[,extracted_var], n = -id_i))
  } else if (id_i >= 0){
    d_all <- d_all %>% mutate(dplyr::lag(d_i[,extracted_var], n = id_i))
  }
  colnames(d_all)[ncol(d_all)] <- sprintf("run%s", i)
}
# Delete temporal objects
rm(i); rm(id_i); rm(d_i); rm(d_all00); rm(d_id_all)

# Trim data
d_all <- d_all[1:(5*256),]
# Standardize temperature
d_all$temperature <- as.numeric(scale(d_all$temperature))
d_all$temperature_input <- as.numeric(scale(d_all$temperature_input))
d_all_std <- na.omit(d_all[(std_trunclen + 1):nrow(d_all),])
d_all_std[,3:ncol(d_all_std)] <- apply(d_all_std[,3:ncol(d_all_std)], 2, function(x) as.numeric(scale(x)))


# <---------------------------------------------> #
#             Visualize patterns
# <---------------------------------------------> #
# Adjust values for visualization
d_all2 <- d_all                  # run1 = mid medium
d_all2$run2 <- d_all2$run2 - 1   # run2 = mid medium
d_all2$run3 <- d_all2$run3 - 3   # run3 = low medium
d_all2$run4 <- d_all2$run4 - 4   # run4 = low medium
d_all2$run5 <- d_all2$run5 - 6   # run5 = high medium
d_all2$run6 <- d_all2$run6 - 7   # run6 = high medium

# Standardized data
d_all3 <- d_all_std; a = 3         # run1 = mid medium
d_all3$run2 <- d_all3$run2 - 1*a   # run2 = mid medium
d_all3$run3 <- d_all3$run3 - 3*a   # run3 = low medium
d_all3$run4 <- d_all3$run4 - 4*a   # run4 = low medium
d_all3$run5 <- d_all3$run5 - 6*a   # run5 = high medium
d_all3$run6 <- d_all3$run6 - 7*a   # run6 = high medium

# Convert data
d_all_long <- d_all2 %>% pivot_longer(cols = -c(time_id, temperature_input, temperature), names_to = "run", values_to = "state")
d_all_long2 <- d_all3 %>% pivot_longer(cols = -c(time_id, temperature_input, temperature), names_to = "run", values_to = "state")

# Visualize with ggplot2
g2 <- ggplot(d_all_long, aes(x = time_id, y = state, color = run)) +
  geom_vline(xintercept = seq(1, nrow(d_all), by = 60), color = "gray", alpha = 0.7, size = 0.3) +
  geom_hline(yintercept = c(0,-1,-3,-4,-6,-7,#-11,#-12,
                            NULL),
             color = "gray", linetype = 2, size = 0.5) +
  geom_line(alpha = 1) + scale_color_igv() + xlab("Time index") + ylab("Reservoir state") +
  theme(axis.text.y = element_blank()) +
  NULL
g3 <- ggplot(d_all_long2, aes(x = time_id, y = state, color = run)) +
  geom_vline(xintercept = seq(d_all_std$time_id[1], rev(d_all_std$time_id)[1], by = 60),
             color = "gray", alpha = 0.7, size = 0.3) +
  geom_hline(yintercept = c(0,-1,-3,-4,-6,-7,#-11,#-12,
                            NULL) * a,
             color = "gray", linetype = 2, size = 0.5) +
  geom_line(alpha = 1) + scale_color_igv() + xlab("Time index") + ylab("Reservoir state") +
  theme(axis.text.y = element_blank()) +
  NULL

# Save figure
ggsave(sprintf("%s/all_dynamics.pdf", output_folder), g2, width = 14, height = 5)
ggsave(sprintf("%s/all_dynamics_std.pdf", output_folder), g3, width = 14, height = 5)


# <---------------------------------------------> #
#         Compile for time-multiplexing
# <---------------------------------------------> #
time_id5 <- seq(1, nrow(d_all), by = 5)
time_id_std5 <- seq(1, nrow(d_all_std), by = 5)
# Define a helper function
demultiplex_run <- function(run_data, time_id_data, old_col = "run1"){
  new_col <- sprintf("%s_%s", old_col, 1:5)
  run_data[time_id_data, all_of(old_col)] %>% rename(!!all_of(new_col)[1] := all_of(old_col)) %>%
    bind_cols(run_data[time_id_data+1, all_of(old_col)]) %>% rename(!!all_of(new_col)[2] := all_of(old_col)) %>%
    bind_cols(run_data[time_id_data+2, all_of(old_col)]) %>% rename(!!all_of(new_col)[3] := all_of(old_col)) %>%
    bind_cols(run_data[time_id_data+3, all_of(old_col)]) %>% rename(!!all_of(new_col)[4] := all_of(old_col)) %>%
    bind_cols(run_data[time_id_data+4, all_of(old_col)]) %>% rename(!!all_of(new_col)[5] := all_of(old_col)) 
}
d_all5 <- d_all[time_id5,1:3] %>%
  mutate(demultiplex_run(d_all, time_id5, old_col = "run1")) %>%
  mutate(demultiplex_run(d_all, time_id5, old_col = "run2")) %>%
  mutate(demultiplex_run(d_all, time_id5, old_col = "run3")) %>%
  mutate(demultiplex_run(d_all, time_id5, old_col = "run4")) %>%
  mutate(demultiplex_run(d_all, time_id5, old_col = "run5")) %>%
  mutate(demultiplex_run(d_all, time_id5, old_col = "run6"))
d_all_std5 <- d_all_std[time_id_std5,1:3] %>%
  mutate(demultiplex_run(d_all_std, time_id_std5, old_col = "run1")) %>%
  mutate(demultiplex_run(d_all_std, time_id_std5, old_col = "run2")) %>%
  mutate(demultiplex_run(d_all_std, time_id_std5, old_col = "run3")) %>%
  mutate(demultiplex_run(d_all_std, time_id_std5, old_col = "run4")) %>%
  mutate(demultiplex_run(d_all_std, time_id_std5, old_col = "run5")) %>%
  mutate(demultiplex_run(d_all_std, time_id_std5, old_col = "run6"))



# <---------------------------------------------> #
#                 Save output
# <---------------------------------------------> #
# Save data
write.csv(na.omit(d_all), sprintf("%s/d_all.csv", output_folder), row.names = F)
write.csv(d_all_std, sprintf("%s/d_std_all.csv", output_folder), row.names = F)
write.csv(na.omit(d_all5), sprintf("%s/d_all5.csv", output_folder), row.names = F)
write.csv(d_all_std5, sprintf("%s/d_std_all5.csv", output_folder), row.names = F)
# Save workspace and objects
save.image(sprintf("%s/%s.RData", output_folder, output_folder))

# Save session info
writeLines(capture.output(sessionInfo()),
           sprintf("00_SessionInfo/%s_SessionInfo_%s.txt", output_folder, output_folder, substr(Sys.time(), 1, 10)))
