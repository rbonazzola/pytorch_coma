METADATA_F <- "data/metadata.csv"
# PARAMS_F <- "data/coma_run_parameters2.tsv"
OUTPUT_DIR <- "data/coma_output2"
PARAMS_F <- file.path(OUTPUT_DIR, "all_run_parameters.tsv")
INDICES_F <- "data/cardiac_indices.csv"
GWAS_DIR <- "data/coma_output/GWAS"

gwas_paths <- list(
  "tsv" = file.path(GWAS_DIR, "{run_id}/GWAS__{latent_variable}.tsv"),
  "region_wise_summary" = gwas_fp_rds <- file.path(GWAS_DIR, "{run_id}/GWAS__{latent_variable}.tsv"),
  "qqplot" = file.path(GWAS_DIR, "{run_id}/GWAS__{latent_variable}__QQ-plot.png"),
  "manhattan" = file.path(GWAS_DIR, "{run_id}/GWAS__{latent_variable}__manhattan.png"),
  "pooled_qqplot" = file.path(GWAS_DIR, "{run_id}/GWAS__all__QQ-plot.png"),
  "rds" = file.path(GWAS_DIR, "{run_id}/GWAS__{latent_variable}.rds")
)

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

relevant_cols <-c("experiment", "kld_weight", "z", "learning_rate", "seed")