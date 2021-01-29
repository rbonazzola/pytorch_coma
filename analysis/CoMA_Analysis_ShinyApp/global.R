library(dplyr)
library(DT)
source("constants.R")
source("utils.R")
source("plots.R")

indices_df <- read.csv(INDICES_F)

params_df <- read.delim(PARAMS_F) 
params_df <- params_df %>% select(-all_of(intersect(columns_to_remove, colnames(params_df))))
assoc_df <- read.csv("data/metadata.csv")
diagnoses_df <- NULL

runs <- vector(mode = "character", length=0)

# Collect all the runs that completed the training.
for (run_id in list.dirs(OUTPUT_DIR, full.names = TRUE, recursive = FALSE)) {
  if (file.exists(file.path(run_id, ".finished")))
    runs <- c(runs, basename(run_id))
}

cardiac_indices <- colnames(indices_df)[grepl(pattern="LV|RV|LA|RA", colnames(indices_df))]
non_cardiac_data <- colnames(assoc_df)[!colnames(assoc_df) %in% cardiac_indices]

assoc_df <- inner_join(assoc_df, indices_df, by="ID")
assoc_df[,cardiac_indices] <- sapply(assoc_df[, cardiac_indices], as.numeric)

col_types <- sapply(assoc_df[-1], class)