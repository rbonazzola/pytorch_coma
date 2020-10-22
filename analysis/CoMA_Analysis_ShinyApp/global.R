library(dplyr)
library(DT)
source("utils.R")
source("plots.R")

OUTPUT_DIR <- "data/coma_output"
indices_df <- read.csv("data/cardiac_indices.csv")

# columns_to_remove <- -run_id, -activation_function, -eval, -data_dir, -eval, -ids_file, -n_layers, -output_dir, -polygon_order, -procrustes_type, -visual_output_dir, -weight_loss, -workers_thread
columns_to_remove <- c("run_id", "activation_function", "eval", "data_dir", "eval", "ids_file", "n_layers", "output_dir", "polygon_order", "procrustes_type", "visual_output_dir", "weight_loss", "workers_thread", "checkpoint_file", "preprocessed_data", "save_all_models", "stop_if_not_learning", "visualize", "template_fname")
params_df <- read.delim("data/coma_run_parameters.tsv") %>% select(-all_of(columns_to_remove))
# params_df %>% group_by(z, kld_weight, learning_rate, procrustes_scaling, partition) %>% summarise(n=n())

metadata_df <- read.csv("data/metadata.csv")
diagnoses_df <- NULL

runs <- vector(mode = "character", length=0)

# Collect all the runs that completed the training.
for (run_id in list.dirs(OUTPUT_DIR, full.names = TRUE, recursive = FALSE)) {
  if (file.exists(file.path(run_id, ".finished")))
    runs <- c(runs, basename(run_id))
}

cardiac_indices <- colnames(indices_df)#[grepl(pattern="LV|RV|LA|RA", colnames(df))]

# df[,cardiac_indices] <- sapply(df[,cardiac_indices], as.numeric)

gwas_paths <- list(
  "qqplot"="",
  "manhattan"="",
  "pooled_qqplot"=""
)

