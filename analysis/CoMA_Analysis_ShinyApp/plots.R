library(ggplot2)

theme_set(theme_bw())

summary_plot <- function(loss) {
 # pp <- ggplot(df, aes_string(x = loss))
 # pp <- pp + geom_histogram()
 # pp <- pp + theme_bw()
 # pp
}

z_density_plot <- function(run_id, z_i, z_j){
  z_df <- get_z_df(run_id)
  # pp <- ggplot(z_df, aes_string(x = input$z_i))
  # pp <- pp + geom_histogram()
  pp <- ggplot(z_df, aes_string(x = z_i, y = z_j))
  #pp <- pp + geom_point()
  pp <- pp + geom_density_2d()
  pp <- pp + theme_bw()
  pp
}

perf_box_plot <- function(run_id) {
  perf_df <- get_perf_df(run_id)
  pp <- ggplot(perf_df %>% filter(mse < quantile(mse, 0.99)), aes(x = as.factor(subset), y = mse))
  pp <- pp + geom_boxplot()
  pp <- pp + theme_bw()
  pp
}


# plot all vs all
z_all_w_all <- function(run_id){
  z_df <- get_z_df(input$run_id)
  pp <- ggplot(z_df, aes_string(x = input$z_i, y = input$z_j))
  pp <- pp + geom_density_2d()
  pp <- pp + theme_bw()
  pp
}

assoc_plot <- function(phenotype_x, phenotype_y) {
  pp <- ggplot(assoc_df, aes_string(x = phenotype_x, y = phenotype_y))
  pp <- pp + theme_bw()
  
  # If variable is categorical use box plot, otherwise scatterplot.
  if (class(assoc_df[,phenotype_x]) == "factor")
    pp <- pp + geom_boxplot()
  else
    pp <- pp + geom_point()
  pp
}

z_animation <- function(){}