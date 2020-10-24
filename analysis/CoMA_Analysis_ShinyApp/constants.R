METADATA_F <- "data/metadata.csv"
PARAMS_F <- "data/coma_run_parameters.tsv"
OUTPUT_DIR <- "data/coma_output"
INDICES_F <- "data/cardiac_indices.csv"

gwas_paths <- list(
  "qqplot"="",
  "manhattan"="",
  "pooled_qqplot"=""
)

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