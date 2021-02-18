METADATA_F <- "data/metadata.csv"
# PARAMS_F <- "data/coma_run_parameters2.tsv"
OUTPUT_DIR <- "data/coma_output"
PARAMS_F <- file.path(OUTPUT_DIR, "all_run_parameters.tsv")
INDICES_F <- "data/cardiac_indices.csv"
GWAS_DIR <- "data/gwas_output"
IDS_DIR <- "data/ids_list"

suffix <- "std_covariates__GBR__qc"

gwas_paths <- list(
  "gwas_hits_summary" = file.path(GWAS_DIR, "{input$run_id}/{suffix}/summaries/GWAS_hits__{suffix}.csv"),
  "region_wise_summary" = file.path(GWAS_DIR, "{input$run_id}/{suffix}/summaries/GWAS__z{as.character(input$z_gwas)}__{suffix}__regionwise_summary.tsv"),
  "qqplot" = file.path(GWAS_DIR, "{input$run_id}/{suffix}/figures/GWAS__z{as.character(input$z_gwas)}__{suffix}__QQ-plot.png"),
  "manhattan" = file.path(GWAS_DIR, "{input$run_id}/{suffix}/figures/GWAS__z{as.character(input$z_gwas)}__{suffix}__manhattan.png"),
  "pooled_qqplot" = file.path(GWAS_DIR, "{input$run_id}/{suffix}/figures/GWAS__all__QQ-plot.png")
  # "tsv" = file.path(GWAS_DIR, "{run_id}/GWAS__{latent_variable}.tsv"),
  # "rds" = file.path(GWAS_DIR, "{run_id}/GWAS__{latent_variable}.rds")
)

median_mse_filename <- "data/coma_output/median_mse_per_run.csv"

# Columns to remove from params_df
columns_to_remove <- c(
  "run_id",
  "activation_function",
  "eval",
  "data_dir",
  "eval",
  "ids_file",
  "n_layers",
  "output_dir",
  "polygon_order",
  "procrustes_type",
  "visual_output_dir",
  "weight_loss",
  "workers_thread",
  "checkpoint_file",
  "preprocessed_data",
  "save_all_models",
  "stop_if_not_learning",
  "visualize",
  "template_fname"
)

# relevant_cols <-c("experiment", "kld_weight", "z", "learning_rate", "seed")
relevant_config_cols <-c("partition", "procrustes_scaling", "z", "kld_weight", "learning_rate", "downsampling_factors")