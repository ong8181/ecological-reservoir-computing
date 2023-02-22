####
#### Assemble figures for ERC paper
#### In silico ERC
####

# Load library
library(tidyverse); packageVersion("tidyverse") # 1.3.2, 2022.11.17
library(cowplot); packageVersion("cowplot") # 1.1.1
library(ggsci); packageVersion("ggsci") # 2.9
library(scales); packageVersion("scales") # 1.2.1, 2022.11.17
library(magick); packageVersion("magick") # 2.7.3, 2022.11.17
options('tibble.print_max' = 20)
theme_set(theme_cowplot())


# <----------------------------------------------------> #
#            Specify input/output directory
# <----------------------------------------------------> #
# Create figure output directory
fig_output_dir <- "0_FormattedFigs"
fig_input_dir1 <- "../01_ERCinsilico/FigCode/0_FigObj"
dir.create(fig_output_dir)


# <----------------------------------------------------> #
#                   Load figure data
# <----------------------------------------------------> #
# Concept illustration
fig_erc_concept <- image_read("0_Illustrations/ERCpaper_Fig1_v9.jpg") 
fig_spmulti <- image_read("0_Illustrations/ERCpaper_Fig2_v3.jpg") 
# Load IPC figures
fig_ipc_rank1 <- image_read("../01_ERCinsilico/FigCode/0_FigObj/IPC_rank1.jpg") 
fig_ipc_rank2 <- image_read("../01_ERCinsilico/FigCode/0_FigObj/IPC_rank2.jpg")
fig_ipc_scatter <- image_read("../01_ERCinsilico/FigCode/0_FigObj/linearity_nonlinearity_MUed.jpg") 
# For echo state network
fig_esn_lorenz <- readRDS(sprintf("%s/Figs_ESN_LorenzPred.obj", fig_input_dir1))
fig_esn_narma <- readRDS(sprintf("%s/Figs_ESN_NARMA.obj", fig_input_dir1))
esn_narma_res <- readRDS(sprintf("%s/ESN_NARMA_results.obj", fig_input_dir1))
fig_esn_mackey <- readRDS(sprintf("%s/Figs_ESN_MackeyGlass.obj", fig_input_dir1))
fig_esn_memory <- readRDS(sprintf("%s/Figs_ESN_MemoryCapacity.obj", fig_input_dir1))
# For ecological reservoir computing
fig_erc_lorenz1 <- readRDS(sprintf("%s/Figs_ERC_LogisticEQN_LorenzPred.obj", fig_input_dir1))
fig_erc_lorenz2 <- readRDS(sprintf("%s/Figs_ERC_LogisticERC_LorenzPred.obj", fig_input_dir1))
fig_erc_lorenz3 <- readRDS(sprintf("%s/Figs_ERC_multiERC_LorenzPred.obj", fig_input_dir1))
fig_erc_narma1 <- readRDS(sprintf("%s/Figs_ERC_NARMA_singleERC.obj", fig_input_dir1))
fig_erc_narma2 <- readRDS(sprintf("%s/Figs_ERC_NARMA_multiERC.obj", fig_input_dir1))
fig_erc_narma3 <- readRDS(sprintf("%s/Figs_ERC_NARMA_multiSummary.obj", fig_input_dir1))
fig_erc_narma4 <- readRDS(sprintf("%s/Figs_ERC_NARMA_MC.obj", fig_input_dir1))
fig_erc_mackey <- readRDS(sprintf("%s/Figs_ERC_MackeyGlass.obj", fig_input_dir1))
fig_erc_memory <- readRDS(sprintf("%s/Figs_ERC_MemoryCapacity.obj", fig_input_dir1))

# Time series example
fig_ts <- readRDS(sprintf("%s/Figs_TimeSeriesExample.obj", fig_input_dir1))

# ESP of ESN and ERC
# ! these files are heavy and not included in the Github repository ! #
# Please contact ong8181@gmail.com if necessary
# Or please reproduce them executing the codes
fig_esp <- readRDS(sprintf("%s/Figs_ESP2_all.obj", fig_input_dir1))


# <----------------------------------------------------> #
#                  Format figures
# <----------------------------------------------------> #
# Figure 1: Concept and Lorenz prediction
## Align panels
Fig_Concept <- ggdraw() + draw_image(fig_erc_concept)
Fig_SpMulti <- ggdraw() + draw_image(fig_spmulti)

# Figure 3: NARMA and Mackey-Glass equation
## Calculate NMSE
nmse <- function(obs, pred){ return(sum((obs-pred)^2)/sum(obs^2)) }
nmse_esn_narma1 <- nmse(esn_narma_res[[1]]$NARMA02, esn_narma_res[[1]]$Emulated)
nmse_esn_narma2 <- nmse(esn_narma_res[[2]]$NARMA03, esn_narma_res[[2]]$Emulated)
nmse_esn_narma3 <- nmse(esn_narma_res[[3]]$NARMA04, esn_narma_res[[3]]$Emulated)
nmse_esn_narma4 <- nmse(esn_narma_res[[4]]$NARMA05, esn_narma_res[[4]]$Emulated)
nmse_esn_narma5 <- nmse(esn_narma_res[[5]]$NARMA10, esn_narma_res[[5]]$Emulated)
nmse_erc_narma1 <- nmse(fig_erc_narma3[[1]]$data$NARMA02, fig_erc_narma3[[1]]$data$Emulated)
nmse_erc_narma2 <- nmse(fig_erc_narma3[[3]]$data$NARMA03, fig_erc_narma3[[3]]$data$Emulated)
nmse_erc_narma3 <- nmse(fig_erc_narma3[[5]]$data$NARMA04, fig_erc_narma3[[5]]$data$Emulated)
nmse_erc_narma4 <- nmse(fig_erc_narma3[[7]]$data$NARMA05, fig_erc_narma3[[7]]$data$Emulated)
nmse_erc_narma5 <- nmse(fig_erc_narma3[[9]]$data$NARMA10, fig_erc_narma3[[9]]$data$Emulated)
nmse_narma_df <- data.frame(task = "NARMA2", method = "ESN", nmse = nmse_esn_narma1) %>%
  rbind(data.frame(task = "NARMA3", method = "ESN", nmse = nmse_esn_narma2)) %>%
  rbind(data.frame(task = "NARMA4", method = "ESN", nmse = nmse_esn_narma3)) %>%
  rbind(data.frame(task = "NARMA5", method = "ESN", nmse = nmse_esn_narma4)) %>%
  rbind(data.frame(task = "NARMA10", method = "ESN", nmse = nmse_esn_narma5)) %>%
  rbind(data.frame(task = "NARMA2", method = "ERC", nmse = nmse_erc_narma1)) %>%
  rbind(data.frame(task = "NARMA3", method = "ERC", nmse = nmse_erc_narma2)) %>%
  rbind(data.frame(task = "NARMA4", method = "ERC", nmse = nmse_erc_narma3)) %>%
  rbind(data.frame(task = "NARMA5", method = "ERC", nmse = nmse_erc_narma4)) %>%
  rbind(data.frame(task = "NARMA10", method = "ERC", nmse = nmse_erc_narma5))
nmse_narma_df$task <- factor(nmse_narma_df$task,
                             levels = c("NARMA2", "NARMA3", "NARMA4", "NARMA5", "NARMA10"))
Fig_NarmaNMSE <- ggplot(nmse_narma_df, aes(x = task, y = nmse, color = method, group = method)) +
  geom_point(aes(shape = method), size = 2) +
  geom_line(linewidth = 0.2) +
  scale_color_manual(values = c("red3", "royalblue")) +
  scale_y_log10(limits = c(1e-04, 1),
                label= function(x) {ifelse(x==1, "1", parse(text=gsub("[+]", "", gsub("e", " %*% 10^", scientific_format()(x)))))}) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  xlab(NULL) + ylab("NMSE") +
  NULL
Fig_NarmaMackey <- plot_grid(fig_erc_lorenz3[[2]] + ggtitle("Lorenz attractor predicted by species-multiplexing ERC") +
                               theme(plot.title = element_text(size = 13, face = "plain")) + xlab("Time step"),
                             fig_erc_lorenz3[[4]],
                             fig_erc_narma3[[2]] + ggtitle("NARMA2 emulated by species-multiplexing ERC") +
                               theme(plot.title = element_text(size = 13, face = "plain")) + xlab("Time step"),
                             Fig_NarmaNMSE,
                             fig_erc_mackey[[2]]  + ggtitle("Mackey-Glass equation embedded by single-species ERC") +
                               theme(plot.title = element_text(size = 13, face = "plain")) + xlab("Time step"),
                             fig_erc_mackey[[3]] + xlab(expression(italic(t))) + ylab(expression(italic(t)-"17")),
                             byrow = T, ncol = 2, rel_widths = c(1.5,1),
                             labels = letters[1:6], align = "hv", axis = "lrtb")


# <----------------------------------------------------> #
#         Supplementary figures
# <----------------------------------------------------> #
# Logistic map reservoir and time series
Fig_Logistic_Lorenz1 <- plot_grid(fig_erc_lorenz1[[1]] + ggtitle(NULL) +
                                    annotate("text", x = 0.25, y = 0.8, label = "RMSE = 0.0897"),
                                  fig_erc_lorenz1[[2]] + ggtitle(NULL) +
                                    annotate("text", x = 0.25, y = 0.8, label = "RMSE = 0.1235"),
                                  ncol = 2, rel_widths = c(1,1), labels = c("c", "d"),
                                  align = "hv")
Fig_Logistic <- plot_grid(plot_grid(fig_ts[[4]] + ylab(expression(italic(X))) + xlab("Time step"),
                                    fig_ts[[1]] + ylab(expression(italic(X))) + xlab("Time step"),
                                    labels = c("a", "b")),
                          Fig_Logistic_Lorenz1,
                          fig_erc_lorenz1[[3]] +
                            xlim(0,70) +
                            geom_point(size = 0.1) +
                            geom_line(size = 0.1) +
                            ggtitle(NULL) +
                            xlab("Time step") +
                            theme(legend.position = "top"),
                          ncol = 1, labels = c(NA, NA, "e"))

# IPC figures
Fig_IPC1 <- ggdraw() + draw_image(fig_ipc_rank1)
Fig_IPC2 <- ggdraw() + draw_image(fig_ipc_rank2)
Fig_IPC3 <- ggdraw() + draw_image(fig_ipc_scatter)
Fig_IPC <- plot_grid(Fig_IPC1, Fig_IPC2, Fig_IPC3,
                     ncol = 1, labels = "auto",
                     rel_heights = c(1,1,2))



# Species-multiplexed in silico ERC
ggsave("esp1_tmp.jpg",
       plot = fig_esp[[1]] +
         ylab("Difference between two states") +
         ggtitle(expression("ESP of fish reservoir")),
       width = 4, height = 4)
esp1_tmp <- image_read("esp1_tmp.jpg") %>% draw_image()
ggsave("esp2_tmp.jpg",
       plot = fig_esp[[2]] +
         ylab("Difference between two states") +
         ggtitle(expression("ESP of prokaryote reservoir")),
       width = 4, height = 4)
esp2_tmp <- image_read("esp2_tmp.jpg") %>% draw_image()
g_esp1 <- ggdraw() + esp1_tmp
g_esp2 <- ggdraw() + esp2_tmp
system("rm esp1_tmp.jpg")
system("rm esp2_tmp.jpg")

Fig_MultiERC_Prop <- plot_grid(g_esp1, g_esp2,
                               fig_erc_memory[[1]] +
                                 ggtitle("Forgetting curve") +
                                 theme(legend.position = "none",
                                       plot.margin = margin(0,1.35,0,0.85,"cm"),
                                       plot.title = element_text(size = 12, face = "plain")),
                               get_legend( fig_erc_memory[[1]] ),
                               ncol = 2, labels = c("b", "c", "d", NA))
Fig_MultiERC <- plot_grid(Fig_SpMulti, Fig_MultiERC_Prop, ncol = 1,
                          labels = c("a", NA), rel_heights = c(0.8,1.1))


# <----------------------------------------------------> #
#                  Save figures
# <----------------------------------------------------> #
ggsave2(sprintf("%s/Fig_AllConcept.pdf", fig_output_dir), Fig_Concept, width = 11, height = 10)
ggsave2(sprintf("%s/Fig_InSilico_NarmaMackey.pdf", fig_output_dir), Fig_NarmaMackey, width = 12, height = 10)
ggsave2(sprintf("%s/Fig_InSilico_IPC.pdf", fig_output_dir), Fig_IPC, width = 8, height = 10)
ggsave2(sprintf("%s/Fig_InSilico_Logistic.pdf", fig_output_dir), Fig_Logistic + theme(plot.margin = unit(c(0,.5,0,.5), "cm")), width = 8, height = 8)

# ! these files are heavy and not included in the Github repository ! #
# Please contact ong8181@gmail.com if necessary
# Or please reproduce them executing the codes
ggsave2(sprintf("%s/Fig_InSilico_MultiERC.pdf", fig_output_dir), Fig_MultiERC, width = 8, height = 10)

