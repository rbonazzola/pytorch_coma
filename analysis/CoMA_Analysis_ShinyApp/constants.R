METADATA_F <- "data/metadata.csv"
PARAMS_F <- "data/coma_run_parameters.tsv"
OUTPUT_DIR <- "data/coma_output"
INDICES_F <- "data/cardiac_indices.csv"
GWAS_DIR <- "data/coma_output/GWAS"

gwas_paths <- list(
  "qqplot"=file.path(GWAS_DIR, "GWAS__{pheno}__QQ-plot.png"),
  "manhattan"=file.path(GWAS_DIR, "GWAS__{pheno}__manhattan.png"),
  "pooled_qqplot"=file.path(GWAS_DIR, "GWAS__all__QQ-plot.png"),
  "rds"=gwas_fp_rds <- file.path(GWAS_DIR, "GWAS__{pheno}.rds")
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