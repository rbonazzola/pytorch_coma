library(tidyverse)

# install.packages("corrplot")
library(corrplot)

root_dir <- system("git rev-parse --show-toplevel", intern = TRUE)

source(file.path(root_dir, "analysis/preprocess_ukbb_data.R"))

# Correlation of latent variables with traditional indices
corr_cardiac_indices <- function(run_id, method="spearman") {
  
  root_dir <- system(command="git rev-parse --show-toplevel", intern=TRUE)

  z_df <- read.csv(glue::glue(file.path(root_dir, "output/{run_id}/latent_space.csv")))

  cardiac_indices <- read.csv(file.path(root_dir, "analysis/data/cardiac_indices/CMR_info_LVRVLARA_11350.csv"))
  cardiac_indices <- cardiac_indices %>% mutate(LVMVR=LVM/LVEDV)
  sphericity <- read.csv(file.path(root_dir, "data/transforms/sphericity.csv"))
  cardiac_indices <- left_join(cardiac_indices, sphericity, by="ID")
  
  cardiac_indices <- cbind(cardiac_indices %>% select(ID, starts_with("LV")), cardiac_indices %>% select(-ID, -starts_with("LV")))
  
  print("Cacho")
  rownames(cardiac_indices) <- cardiac_indices$ID
  cardiac_indices <- cardiac_indices[as.character(z_df$ID), ]

  cor(
  	cardiac_indices %>% select(-ID), 
  	z_df %>% select(starts_with("z")), 
  	method=method, use="complete.obs"
  )

}


corr_demographic_data <- function(run_id, method="spearman") {
  
  root_dir <- system(command="git rev-parse --show-toplevel", intern=TRUE)

  field_mapping_ <- read.csv(file.path(root_dir, "data/field_mappings.csv"), stringsAsFactors = F)
  field_mapping <- field_mapping_$description
  names(field_mapping) <- paste0("X", field_mapping_$field_id)
  field_mapping['ID'] <- 'ID'
  
  z_df <- read.csv(file.path(root_dir, glue::glue("output/{run_id}/latent_space.csv")))
  
  demographic_data <- read.csv(file.path(root_dir, "data/datasets/demographic_data.csv"))
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
  
  root_dir <- system(command="git rev-parse --show-toplevel", intern=TRUE)

  z_df1 <- read.csv(glue::glue(file.path(root_dir, "output/{runid1}/latent_space.csv")))
  rownames(z_df1) <- z_df1$ID
  
  z_df2 <- read.csv(glue::glue(file.path(root_dir, "output/{runid2}/latent_space.csv")))
  rownames(z_df2) <- z_df2$ID
 
  z_df2 <- z_df2[as.character(z_df1$ID),]
     
  cor(
  	z_df1 %>% select(-ID) %>% select(starts_with("z")), 
  	z_df2 %>% select(-ID) %>% select(starts_with("z")),
  	method=method, use="complete.obs"
  )

}

within_experiment_corr <- function(runid, method="spearman") {
  
  root_dir <- system(command="git rev-parse --show-toplevel", intern=TRUE)
  
  z_df <- read.csv(glue::glue(file.path(root_dir, "output/{runid}/latent_space.csv")))
  rownames(z_df) <- z_df$ID
  
  cor(
    z_df %>% select(-ID) %>% select(starts_with("z")), 
    method=method, use="complete.obs"
  )
  
}


t_test_icd10 <- function(run_id, pval_as_log10=TRUE, logp_thres=ifelse(pval_as_log10, 8, NULL)) {
  
  # Returns a dataframe where 
  # - the rows are diseases, 
  # - the columns are latent variables, and
  # - the cells contain the t-test p-values
  
  root_dir <- system(command="git rev-parse --show-toplevel", intern=TRUE)
  
  # Load ICD10 code mappings
  mapping_df <- read_tsv(file.path(root_dir, "data/ICD10_code_mapping.tsv"))
  mapping <- mapping_df$meaning
  names(mapping) <- mapping_df$coding
  
  # Load individuals with different diseases
  icd10_dir <- file.path(root_dir, "data/datasets/ids/icd10")
  disease <- lapply(list.files(icd10_dir, full.names = T), readLines)
  names(disease) <- sapply(list.files(icd10_dir), function(x) gsub("\\.txt", "", x))
  
  latent_space <- file.path(root_dir, "output/{run_id}/latent_space.csv")
  health_outcome_png <- file.path(root_dir, "output/{run_id}/health_outcome_t-test.png")
  health_outcome_csv <- file.path(root_dir, "output/{run_id}/health_outcome_t-test.csv")
  
  z_df <- read.csv(glue::glue(latent_space)) %>% select(-subset)
  z_df <- z_df %>% mutate(ID=as.character(ID))
  latvar_names <- colnames(z_df)[grepl(pattern = "z", x = colnames(z_df))]
  
  df <- data.frame(matrix(data = NA, nrow = length(disease), length(latvar_names)))
  colnames(df) <- latvar_names
  rownames(df) <- names(disease)
  ncases <- integer(length = 0)
  
  for(icd10 in names(disease)) {
    icd10_ids <- (z_df %>% filter(ID %in% disease[[icd10]]))$ID
    
    ncases <- c(ncases, length(icd10_ids))
    if (length(icd10_ids) > 20) {
      for (latvar in latvar_names) {
        test <- t.test(
          (z_df %>% filter(ID %in% disease[[icd10]]))[,latvar], 
          (z_df %>% filter(!ID %in% disease[[icd10]]))[,latvar]
        )
        df[icd10, latvar] <- test$p.value
      }
    }
  }
  
  names(ncases) <- names(disease)
  
  df <- na.omit(df)
  rownames(df) <- sapply(rownames(df), function(x) paste0(mapping[x], " (", ncases[x], ")"))
  if (pval_as_log10) {
    df_ <- -log10(df) 
    if (!is.null(logp_thres))
      df_[df_ > logp_thres] <- logp_thres
  }
  
  
  return(df_)
}

# reorder <- corrMatOrder(cross_experiment("2020-09-30_12-36-48", "2020-09-11_02-13-41"), order = "hclust", hclust.method = "ward.D2")