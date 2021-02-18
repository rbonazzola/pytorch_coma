get_z_df <- function(run_id) {
  file <- file.path(OUTPUT_DIR, run_id, "latent_space.csv")
  read.csv(file)
}

get_perf_df <- function(run_id) {
  file <- file.path(OUTPUT_DIR, run_id, "performance.csv")
  read.csv(file)
}

read_perf <- function(run_id, add_run_id = FALSE) { 
  path_to_perf <- file.path(OUTPUT_DIR, run_id, "performance.csv")
  if (file.exists(path_to_perf)) {
    df <- read.csv(path_to_perf)
    if (add_run_id)
      df$run_id <- run_id
    return(df)
  } else {
    return(NULL)
  }
}

gather_perf <- function(runs) {
  names(runs) <- runs
  kk <- lapply(runs, read_perf, TRUE)
  perf_all_df <- bind_rows(kk)
  perf_all_df
}