source("../relation_to_other_variables.R")

function(input, output) {
  
  ### UI: SIDE BAR PANEL ###
  
  # EXPERIMENTAL-LEVEL RESULTS
  output$Results <- renderUI({
    fluidPage(
      titlePanel("Results"),
      selectInput("run_id", "Select experiment", runs),
      
      radioButtons(
        inputId = "ExperimentPlotType",
        choices = c(
          "Performance",
          "Statistical properties",
          "Associations",
          "GWAS"
        ),
        label = "Select type of analysis"
      ),
      
      # STATISTICAL PROPERTIES
      conditionalPanel(
        condition = "input.ExperimentPlotType == \"Statistical properties\"",
        #TODO: Make the range of z's experiment-dependent
        selectInput("z_i", "Select latent variable 1", paste0("z", 0:15)),
        selectInput("z_j", "Select latent variable 2", paste0("z", 0:15))
      ),
      
      # ASSOCIATIONS
      conditionalPanel(
        condition = "input.ExperimentPlotType == \"Associations\"",
        radioButtons(
          inputId = "variableType", 
          choices = c(
            "Cardiac indices", 
            "Other variables", 
            "Diagnoses"
          ), 
          label="Select type of variable"
        ),
        selectInput("latent_variable", "Select latent variable", paste0("z", 0:15), selected = "z0"),
        conditionalPanel(condition = "input.variableType == \"Cardiac indices\"",
          selectInput("yvar1", "Select column", cardiac_indices, selected = "LVEDV")
        ),
        conditionalPanel(condition = "input.variableType == \"Other variables\"",
          selectInput("yvar2", "Select column", non_cardiac_data, selected = "sex")
        )
      ),
      
      # ASSOCIATIONS
      conditionalPanel(
        condition = "input.ExperimentPlotType == \"GWAS\"",
        radioButtons(
          inputId = "GWASPlotType", 
          choices = c(
            "Q-Q plot (all z's pooled)", 
            "Manhattan plot"
          ), 
          label="Select type of plot"
        ),
        sliderInput(
          inputId = "z_gwas", 
          label = "Select latent variable",  
          min = 0, max = 15, value = 0
        )
      )
    )
  })
  
  output$performance_plot <- renderPlot(perf_box_plot(input$run_id))
  
  output$z_scatter_plot <- renderPlot(z_plot(input$run_id, input$z_i, input$z_j))
  
  output$association_plot <- renderPlot(
    switch(
      input$variableType,
      # "Cardiac indices" = { assoc_plot(input$run_id, input$latent_variable, input$yvar1) },
      # "Other variables" = { assoc_plot(input$run_id, input$latent_variable, input$yvar2) }
      "Cardiac indices" = { 
        corrplot::corrplot(corr_cardiac_indices(input$run_id)) 
      },
      "Other variables" = { 
        corrplot::corrplot(corr_demographic_data(input$run_id)) 
      },
      "Diagnoses" = { 
        df <- t_test_icd10(input$run_id)
        max_per_row <- apply(df, 1, FUN=max)
        df <- df[max_per_row > 2,]
        corrplot::corrplot(as.matrix(df), is.corr = FALSE)
      }
    )
    
  )
  
  observe({print(glue::glue(gwas_paths$pooled_qqplot))})
  observe({print(glue::glue(gwas_paths$manhattan))})
  
  output$config_for_this_run <- renderDataTable(
    params_df %>% filter(experiment == input$run_id) %>% select(relevant_config_cols, median_mse)
  )
  
  output$gwas_hits_summary <- renderDataTable(
    read.csv(glue::glue(gwas_paths$gwas_hits_summary)) %>% mutate(P=-log10(P)) 
  )
  
  output$manhattan <- renderImage({ 
    list(
      src = glue::glue(gwas_paths$manhattan),
      width = 900, height = 300
    )}, 
    deleteFile=FALSE 
  )
  
  output$qqplot <- renderImage( { 
    list(
      src = glue::glue(gwas_paths$pooled_qqplot),
      width = 600, height = 600
    ) 
  }, deleteFile=FALSE )
  
  output$params_df <-  DT::renderDT(
    # https://yihui.shinyapps.io/DT-selection/
    DT::datatable(
      params_df, 
      filter = "top", 
      options = list(paging=FALSE)
    ), server=FALSE # server=FALSE to enable row selection
  )
  
  
}

### OUTPUT ###

# observe(
#   {print(input$controlPanel)}
# )

# output$plot <- renderPlot({
#   switch(
#     input$controlPanel,
#     "summaries" = summary_plot(input$which_loss),
#     "experiment_details" = switch(
#       input$ExperimentPlotType,
#       "Performance" = perf_box_plot(input$run_id), 
#       "Statistical properties" = z_plot(input$run_id, input$z_i, input$z_j), 
#       "Associations" = assoc_plot(input$run_id, input$latent_variable, input$yvar)#, 
#       # "GWAS" = 
#     )
#     #, "Docs"
#   )
# })

#TOFIX: Error in $.shinyoutput: Reading from shinyoutput object is not allowed.
# output$brush_info_z <-  DT::renderDataTable({
#   brushedPoints(output$plot, input$plot1_brush) # %>% select(-cardiac_indices)
# })

#output$qqplot_pooled <- renderImage({
#  NULL 
#}, deleteFile = TRUE)
#
# output$manhattan <- renderImage({
#   NULL 
# }, deleteFile = TRUE)

# Reactive values
# rv <- reactiveValues(data = NULL)

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