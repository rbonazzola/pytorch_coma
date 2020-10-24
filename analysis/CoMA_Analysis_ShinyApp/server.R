function(input, output) {
  
  ### UI: SIDE BAR PANEL ###
  
  output$SummTab <- renderUI({
    fluidPage(
      DT::dataTableOutput("params_df")
    )
  })
  
  # output$PerfTab <- renderUI({
  #   fluidPage(
  #     titlePanel("CoMA - performance"),
  #     selectInput("run_id_", "Select experiment", runs),
  #   )
  # })
  
  output$Experiment <- renderUI({
    fluidPage(
      titlePanel("CoMA"),
      selectInput("run_id", "Select experiment", runs),
      
      radioButtons(
        inputId = "ExperimentPlotType", 
        choices = c("Performance", "Statistical properties", "Associations", "GWAS"),
        label = "Select type of analysis"
      ),
      
      # STATISTICAL PROPERTIES
      
      
      # STATISTICAL PROPERTIES
      conditionalPanel(condition = "input.ExperimentPlotType == \"Statistical properties\"",
        #TODO: Make the range of z's experiment-dependent
        selectInput("z_i", "Select latent variable 1", paste0("z", 0:15)),
        selectInput("z_j", "Select latent variable 2", paste0("z", 0:15))
      ),
      
      # ASSOCIATIONS
      conditionalPanel(condition = "input.ExperimentPlotType == \"Associations\"",
        radioButtons(
          inputId = "variableType", 
          choices = c("Cardiac indices", "Other variables", "Diagnoses"), 
          label="Select type of variable"
        ),
        selectInput("latent_variable", "Select variable 1", paste0("z", 0:15), selected = "LVESV"),
        conditionalPanel(condition = "input.variableType == \"Cardiac indices\"",
          selectInput("yvar", "Select column", cardiac_indices, selected = "LVEDV")
        ),
        conditionalPanel(condition = "input.variableType == \"Other variables\"",
          selectInput("yvar", "Select column", non_cardiac_data, selected = "sex")
        )
      ),
      
    )
  })
  
  output$gwas <- renderUI({
    fluidPage(
      titlePanel("GWAS")
    )
  })
  
  output$RunDetails <- renderUI({
    
  })
  
  
  ### OUTPUT ###

  output$plot <- renderPlot({
    switch(
      input$controlPanel,
      "summaries" = summary_plot(input$which_loss),
      "experiment_details" = switch(
        input$ExperimentPlotType,
        "Performance" = perf_box_plot(input$run_id), 
        "Statistical properties" = z_density_plot(input$run_id, input$z_i, input$z_j), 
        "Associations" = assoc_plot(input$run_id, input$latent_variable, input$yvar), 
        # "GWAS" = 
      )
      #, "Docs"
    )
  })
  
  #TOFIX: Error in $.shinyoutput: Reading from shinyoutput object is not allowed.
  # output$brush_info_z <-  DT::renderDataTable({
  #   brushedPoints(output$plot, input$plot1_brush) # %>% select(-cardiac_indices)
  # })
  
  output$qqplot_pooled <- renderImage({
    NULL 
  }, deleteFile = TRUE)
  
  output$manhattan <- renderImage({
    NULL 
  }, deleteFile = TRUE)
  
  # Reactive values
  rv <- reactiveValues(data = NULL)
  output$params_df <-  DT::renderDT(
    # https://yihui.shinyapps.io/DT-selection/
    DT::datatable(params_df, filter = "top", options=list(paging=FALSE)), server=FALSE # server=FALSE to enable row selection
  )
  
  #output$brush_info_cardiac <-  DT::renderDataTable({
  #  brushedPoints(df, input$plot1_brush)[, c("ID", cardiac_indices)]
  #})
  
  # output$brush_info_others <-  DT::renderDataTable({
  #   brushedPoints(df, input$plot1_brush) %>% select(-cardiac_indices)
  # })
  
  
  #observe({
  #  rv$row_indices = rv$params_df(input$params_df_rows_selected)
  #  print(rv$row_indices)
  #})
  #
  
  
  # output$click_info <- renderPrint({
  #   # Because it's a ggplot2, we don't need to supply xvar or yvar; if this
  #   # were a base graphics plot, we'd need those.
  #   nearPoints(df, input$plot1_click, addDist = TRUE)
  # })
  #
  
}