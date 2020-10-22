function(input, output) {
  
  ### UI ###
  
  output$SummTab <- renderUI({
    fluidPage(
      DT::dataTableOutput("params_df")
    )
  })
  
  output$PerfTab <- renderUI({
    fluidPage(
      titlePanel("CoMA - performance"),
      selectInput("run_id_", "Select experiment", runs),
    )
  })
  
  output$ZTab <- renderUI({
    fluidPage(
      titlePanel("CoMA - Latent space"),
      selectInput("run_id", "Select experiment", runs),
      selectInput("z_i", "Select latent variable 1", paste0("z", 0:7)),
      selectInput("z_j", "Select latent variable 2", paste0("z", 0:7))
    )
  })
  
  output$assoc <- renderUI({
    fluidPage(
      titlePanel("X vs. Y"),
      selectInput("phenotype_x", "Select variable 1", names(col_types)[col_types != "integer"], selected = "LVEDV"),
      selectInput("phenotype_y", "Select variable 2", names(col_types), selected = "LVESV")
    )
  })
  
  output$gwas <- renderUI({
    fluidPage(
      titlePanel("GWAS")
    )
  })
  
  
  ############

  output$plot <- renderPlot({
    switch(
      input$controlPanel,
      "summaries" = summary_plot(input$which_loss),
      "latent_space" = z_density_plot(input$run_id, input$z_i, input$z_j),
      "performance" = perf_box_plot(input$run_id_),
      "assoc" = assoc_plot(input$phenotype_x, input$phenotype_y)
      #, "Docs"
    )
  })
  
  output$brush_info_z <-  DT::renderDataTable({
    brushedPoints(output$plot, input$plot1_brush) # %>% select(-cardiac_indices)
  })
  
  #output$brush_info_cardiac <-  DT::renderDataTable({
  #  brushedPoints(df, input$plot1_brush)[, c("ID", cardiac_indices)]
  #})
  
  # output$brush_info_others <-  DT::renderDataTable({
  #   brushedPoints(df, input$plot1_brush) %>% select(-cardiac_indices)
  # })
  
  # Reactive values
  rv <- reactiveValues(data = NULL)
  rv$params_df <-  DT::renderDataTable(
    # https://yihui.shinyapps.io/DT-selection/
    DT::datatable(params_df, options=list(paging=FALSE)), server=FALSE # server=FALSE to enable row selection
  )
  rv$row_indices = rv$params_df(input$params_df_rows_selected)
  print(rv$row_indices)
  
  # output$hist <- renderPlot({
  #   pp <- ggplot(df, aes_string(x = input$phenotype))
  #   pp <- pp + geom_histogram()
  #   pp <- pp + theme_bw()
  #   pp
  # })
  # 
  # output$scatter <- renderPlot({
  #   pp <- ggplot(df, aes_string(x = input$phenotype_x, y = input$phenotype_y))
  #   pp <- pp + theme_bw()
  #   
  #   # If variable is categorical use box plot, otherwise scatterplot.
  #   if (class(df[,input$phenotype_x]) == "factor")
  #     pp <- pp + geom_boxplot() 
  #   else
  #     pp <- pp + geom_point() 
  #   pp
  # })
  
  # output$click_info <- renderPrint({
  #   # Because it's a ggplot2, we don't need to supply xvar or yvar; if this
  #   # were a base graphics plot, we'd need those.
  #   nearPoints(df, input$plot1_click, addDist = TRUE)
  # })
  #
  
}

# input$