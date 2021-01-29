library(tidyverse)

# install.packages("corrplot")
library(corrplot)

source("analysis/preprocess_ukbb_data.R")

# Correlation of latent variables with traditional indices
corr_cardiac_indices <- function(run_id, method="spearman") {
  
  setwd(system(command="git rev-parse --show-toplevel", intern=TRUE))

  z_df <- read.csv(glue::glue("output/{run_id}/latent_space.csv"))

  cardiac_indices <- read.csv("analysis/data/cardiac_indices/CMR_info_LVRVLARA_11350.csv")
  cardiac_indices <- cardiac_indices %>% mutate(LVMVR=LVM/LVEDV)
  sphericity <- read.csv("data/transforms/sphericity.csv")
  cardiac_indices <- left_join(cardiac_indices, sphericity, by="ID")
  
  cardiac_indices <- cbind(cardiac_indices %>% select(ID, starts_with("LV")), cardiac_indices %>% select(-ID, -starts_with("LV")))
  
  
  rownames(cardiac_indices) <- cardiac_indices$ID
  cardiac_indices <- cardiac_indices[as.character(z_df$ID), ]

  cor(
  	cardiac_indices %>% select(-ID), 
  	z_df %>% select(starts_with("z")), 
  	method=method, use="complete.obs"
  )

}


corr_demographic_data <- function(run_id, method="spearman") {
  
  setwd(system(command="git rev-parse --show-toplevel", intern=TRUE))

  field_mapping_ <- read.csv("data/field_mappings.csv", stringsAsFactors = F)
  field_mapping <- field_mapping_$description
  names(field_mapping) <- paste0("X", field_mapping_$field_id)
  field_mapping['ID'] <- 'ID'
  
  z_df <- read.csv(glue::glue("output/{run_id}/latent_space.csv"))
  
  demographic_data <- read.csv("data/datasets/demographic_data.csv")
  demographic_data <- demographic_data %>% rename(ID=eid)
  summarised_cols <- unique(sapply(strsplit(colnames(demographic_data %>% select(-ID)),split = "\\."), function(x) x[1]))
  
  demographic_data <- mean_across_visits(
    demographic_data, 
    summarised_cols, 
    colnames(demographic_data)[startsWith(colnames(demographic_data), "X")]
  )
  # demographic_data <- impute_na(covariates_df, c("X4079", "X4089"))
  # demographic_data <- na.omit(covariates_df)
  # covariates_df <- select(covariates_df, c("ID", covariate_names))
  
  # demographic_data <- demographic_data %>% mutate(LVMVR=LVM/LVEDV)
  # demographic_data <- cbind(demographic_data %>% select(ID, starts_with("LV")), demographic_data %>% select(-ID, -starts_with("LV")))
  
  rownames(demographic_data) <- demographic_data$ID
  demographic_data <- demographic_data[as.character(z_df$ID), ]
  
  colnames(demographic_data) <- sapply(
    colnames(demographic_data), 
    function(field_id) field_mapping[field_id]
  )
  
  demographic_data <- demographic_data[,colnames(demographic_data)[!is.na(colnames(demographic_data))]]
  
  cor(
    demographic_data %>% select(-ID), 
    z_df %>% select(starts_with("z")), 
    method=method, use="complete.obs"
  )
  
}


cross_experiment_corr <- function(runid1, runid2, method="spearman") {
  
  setwd(
    system(command="git rev-parse --show-toplevel", intern=TRUE)
  )

  z_df1 <- read.csv(glue::glue("output/{runid1}/latent_space.csv"))
  rownames(z_df1) <- z_df1$ID
  
  z_df2 <- read.csv(glue::glue("output/{runid2}/latent_space.csv"))
  rownames(z_df2) <- z_df2$ID
 
  z_df2 <- z_df2[as.character(z_df1$ID),]
     
  cor(
  	z_df1 %>% select(-ID) %>% select(starts_with("z")), 
  	z_df2 %>% select(-ID) %>% select(starts_with("z")),
  	method=method, use="complete.obs"
  )

}

within_experiment_corr <- function(runid, method="spearman") {
  
  setwd(system(command="git rev-parse --show-toplevel", intern=TRUE))
  
  z_df <- read.csv(glue::glue("output/{runid}/latent_space.csv"))
  rownames(z_df) <- z_df$ID
  
  cor(
    z_df %>% select(-ID) %>% select(starts_with("z")), 
    method=method, use="complete.obs"
  )
  
}


t_test_icd10 <- function(run_id) {
  
  # Returns a dataframe where 
  # - the rows are diseases, 
  # - the columns are latent variables, and
  # - the cells contain the t-test p-values
  
  # Load ICD10 code mappings
  mapping_df <- read_tsv("data/ICD10_code_mapping.tsv")
  mapping <- mapping_df$meaning
  names(mapping) <- mapping_df$coding
  
  # Load individuals with different diseases
  icd10_dir <- "data/datasets/ids/icd10"
  disease <- lapply(list.files(icd10_dir, full.names = T), readLines)
  names(disease) <- sapply(list.files(icd10_dir), function(x) gsub("\\.txt", "", x))
  
  latent_space <- "output/{run_id}/latent_space.csv"
  health_outcome_png <- "output/{run_id}/health_outcome_t-test.png"
  health_outcome_csv <- "output/{run_id}/health_outcome_t-test.csv"
  
  z_df <- read.csv(glue::glue(latent_space)) %>% select(-subset)
  z_df <- z_df %>% mutate(ID=as.character(ID))
  latvar_names <- colnames(z_df)[grepl(pattern = "z", x = colnames(z_df))]
  
  df <- data.frame(matrix(data = NA, nrow = length(disease), length(latvar_names)))
  colnames(df) <- latvar_names
  rownames(df) <- names(disease)
  
  for (latvar in latvar_names) {
    for(icd10 in names(disease)) {
      icd10_ids <- (z_df %>% filter(ID %in% disease[[icd10]]))$ID
      if (length(icd10_ids) > 20) {
        test <- t.test(
          (z_df %>% filter(ID %in% disease[[icd10]]))[,latvar], 
          (z_df %>% filter(!ID %in% disease[[icd10]]))[,latvar]
        )
        df[icd10,latvar] <- test$p.value
      }
    }
  }
  
  df <- na.omit(df)
  rownames(df) <- sapply(rownames(df), function(x) mapping[x])
  df_ <- -log10(df)
  df_[df_ > 8] <- 8
  # rownames(df_) <- sapply(rownames(df_), function(x) mapping[x])
  
  png(filename = glue::glue(health_outcome_png), width=800, height=2100)
  corrplot::corrplot(as.matrix(df_), is.corr = FALSE, method = "number")
  dev.off()
  
  write_csv(df, glue::glue(health_outcome_csv))
  
  return(df)
}

# reorder <- corrMatOrder(cross_experiment("2020-09-30_12-36-48", "2020-09-11_02-13-41"), order = "hclust", hclust.method = "ward.D2")