source("relation_to_other_variables.R")

run_ids <- c(
  "experiment_1"="2020-09-11_02-13-41", 
  "experiment_2"="2020-09-30_12-36-48"
)

generate_health_outcomes_fig <- function(run_id) {
  df <- t_test_icd10(run_id, pval_as_log10=TRUE, logp_thres=8)
  max_per_row <- apply(df, 1, FUN=max)
  # print(min_per_row)
  df <- df[max_per_row > 2,]
  # print(df)
  pp <- corrplot::corrplot(as.matrix(df), is.corr = FALSE, method = "number")
  pp
}


generate_figures <- function(run_id, alias=run_id) {
  root_dir <- system(command="git rev-parse --show-toplevel", intern=TRUE)
    
  png(file.path(root_dir, glue::glue("output/{run_id}/{alias}_vs_cardiac_indices.png")), res = 100)
  pp <- corrplot::corrplot(corr_cardiac_indices(run_id))
  print(pp)
  dev.off()

  png(file.path(root_dir, glue::glue("output/{run_id}/{alias}_vs_demographic_data.png")), width=600, height=300)
  pp <- corrplot::corrplot(corr_demographic_data(run_id))
  print(pp)
  dev.off()
  
  png(file.path(root_dir, glue::glue("output/{run_id}/{alias}_health_outcome_t-test.png")), width=900, height=1200)
  generate_health_outcomes_fig(run_id)
  dev.off()
}

# CORRELATION WITH CARDIAC INDICES
for (i in seq_along(run_ids)) {
  run_id <- run_ids[i]
  alias <- names(run_ids)[i]
  generate_figures(run_id, alias)
}
