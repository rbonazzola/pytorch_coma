library(tidyverse)

# install.packages("corrplot")
library(corrplot)

source("analysis/preprocess_ukbb_data.R")


# Correlation of latent variables with traditional indices
corr_cardiac_indices <- function(run_id, method="spearman") {
  
  setwd(
    system(command="git rev-parse --show-toplevel", intern=TRUE)
  )

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
  
  setwd(
    system(command="git rev-parse --show-toplevel", intern=TRUE)
  )
  
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
  
  setwd(
    system(command="git rev-parse --show-toplevel", intern=TRUE)
  )
  
  z_df <- read.csv(glue::glue("output/{runid}/latent_space.csv"))
  rownames(z_df) <- z_df$ID
  
  cor(
    z_df %>% select(-ID) %>% select(starts_with("z")), 
    method=method, use="complete.obs"
  )
  
}

# reorder <- corrMatOrder(cross_experiment("2020-09-30_12-36-48", "2020-09-11_02-13-41"), order = "hclust", hclust.method = "ward.D2")