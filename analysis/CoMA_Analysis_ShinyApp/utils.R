get_z_df <- function(run_id) {
  file <- file.path(OUTPUT_DIR, run_id, "latent_space.csv")
  read.csv(file)
}

get_perf_df <- function(run_id) {
  file <- file.path(OUTPUT_DIR, run_id, "performance.csv")
  read.csv(file)
}