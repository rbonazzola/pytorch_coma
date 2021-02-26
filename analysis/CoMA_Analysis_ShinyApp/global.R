library(tidyverse)
library(DT)
source("constants.R")
source("utils.R")
source("plots.R")

indices_df <- read.csv(INDICES_F)

params_df <- read.delim(PARAMS_F) 
params_df <- params_df %>% select(-all_of(intersect(columns_to_remove, colnames(params_df))))
params_df <- cbind(params_df %>% select(relevant_config_cols), params_df %>% select(-all_of(relevant_config_cols)))

rownames(params_df) <- params_df$experiment
params_df <- params_df %>% mutate(kld_weight=-log10(kld_weight))

median_mse <- read_csv(median_mse_filename)
print(str(median_mse))
params_df <- inner_join(params_df, median_mse, by="experiment")

assoc_df <- read.csv("data/metadata.csv")
diagnoses_df <- NULL

runs <- params_df %>% filter(median_mse < 0.7) %>% .$experiment

# runs <- vector(mode = "character", length=0)
# 
# # Collect all the runs that completed the training.
# for (run_id in list.dirs(OUTPUT_DIR, full.names = TRUE, recursive = FALSE)) {
#   if (file.exists(file.path(run_id, ".finished")))
#     runs <- c(runs, basename(run_id))
# }

cardiac_indices <- colnames(indices_df)[grepl(pattern="LV|RV|LA|RA", colnames(indices_df))]
non_cardiac_data <- colnames(assoc_df)[!colnames(assoc_df) %in% cardiac_indices]
assoc_df <- inner_join(assoc_df, indices_df, by="ID")
assoc_df[,cardiac_indices] <- sapply(assoc_df[, cardiac_indices], as.numeric)

col_types <- sapply(assoc_df[-1], class)