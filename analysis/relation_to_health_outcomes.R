source("analysis/relation_to_other_variables.R")

cor(x=cardiac_indices, y=z_df %>% select(starts_with("z")), method="spearman", use="complete.obs")

runid1="2020-09-30_12-36-48"; runid2= "2020-09-11_02-13-41"

z_df <- read.csv("~/PhD/repos/pytorch_coma/output/2020-09-30_12-36-48/latent_space.csv")

z_df %>% filter(complete.cases(.)) %>% sample_n(25) %>% summary
z_df %>% filter(ID %in% dcm & complete.cases(.)) %>% summary

cross_experiment("2020-09-30_12-36-48", "2020-09-11_02-13-41")