library(dplyr)
library(DT)
source("constants.R")
source("utils.R")
source("plots.R")

OUTPUT_DIR <- "data/coma_output"
indices_df <- read.csv("data/cardiac_indices.csv")

params_df <- read.delim("data/coma_run_parameters.tsv") %>% select(-all_of(columns_to_remove))
assoc_df <- read.csv("data/metadata.csv")
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

cardiac_indices <- colnames(assoc_df)[grepl(pattern="LV|RV|LA|RA", colnames(assoc_df))]
assoc_df[,cardiac_indices] <- sapply(assoc_df[, cardiac_indices], as.numeric)
col_types <- sapply(assoc_df[-1], class)

# df[,cardiac_indices] <- sapply(df[,cardiac_indices], as.numeric)

gwas_paths <- list(
  "qqplot"="",
  "manhattan"="",
  "pooled_qqplot"=""
)

