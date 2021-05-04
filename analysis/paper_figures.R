source("analysis/relation_to_other_variables.R")

run_params <- read_yaml("analysis/paper_z_mapping_parameters.yaml")

generate_health_outcomes_fig <- function(run_id, mapping, sign, label) {
  df <- t_test_icd10(run_id, pval_as_log10=TRUE, logp_thres=8, mapping, sign)
  max_per_row <- apply(df, MARGIN = 1, FUN=max)
  
  # print(min_per_row)
  
  df <- df[max_per_row > 2,]
  
  # print(df)
  ttest_mat <- as.matrix(df)
  colnames(ttest_mat) <- label # c(":z[1]^(s)", ":z[2]^(s)", ":z[3]^(s)", ":z[4]^(s)", ":z[5]^(s)", ":z[6]^(s)", ":z[7]^(s)", ":z[8]^(s)")
  pp <- corrplot::corrplot(ttest_mat, is.corr = FALSE, method = "number", tl.srt = 90)
  pp
}


generate_figures <- function(run_id, alias=run_id, mapping, sign, label) {
  root_dir <- system(command="git rev-parse --show-toplevel", intern=TRUE)
    
  png(file.path(root_dir, glue::glue("output/{run_id}/{alias}_vs_cardiac_indices.png")), res = 100)
  corrmat <- corr_cardiac_indices(run_id, mapping, sign)
  colnames(corrmat) <- label #(":z[1]", ":z[2]", ":z[3]", ":z[4]", ":z[5]", ":z[6]", ":z[7]", ":z[8]")
  pp <- corrplot::corrplot(corrmat, tl.srt = 90)
  print(pp)
  dev.off()

  png(file.path(root_dir, glue::glue("output/{run_id}/{alias}_vs_demographic_data.png")), width=600, height=300)
  corrmat <- corr_demographic_data(run_id, mapping, sign)
  colnames(corrmat) <- label # c(":z[1]", ":z[2]", ":z[3]", ":z[4]", ":z[5]", ":z[6]", ":z[7]", ":z[8]")
  pp <- corrplot::corrplot(corrmat, tl.srt = 90)
  # print(pp)
  dev.off()
  
  png(file.path(root_dir, glue::glue("output/{run_id}/{alias}_health_outcome_t-test.png")), width=900, height=1200)
  generate_health_outcomes_fig(run_id, mapping, sign, label)
  dev.off()
}

# CORRELATION WITH CARDIAC INDICES
for (run_id in names(run_params)) {
  alias <- run_params[[run_id]]$alias
  mapping <- run_params[[run_id]]$mapping
  sign <- run_params[[run_id]]$sign
  label <- run_params[[run_id]]$label
  generate_figures(run_id, alias, mapping, sign, label)
}
