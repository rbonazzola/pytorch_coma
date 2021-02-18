shinyUI(
  
  fluidPage(
    
    includeHTML("www/coma.html"),  
    # titlePanel("CoMA Results Explorer"),
    
    sidebarPanel(
      tabsetPanel(
        id = "controlPanel",
        tabPanel(title = "Results", uiOutput("Results"), value = "results"),
        tabPanel(title = "Run Details", value = "run_details"),
        tabPanel(title = "Docs", htmlOutput("docsTab"), value = "docs"),
        type="pills"
      )
    ),
  
    mainPanel(
      fluidRow(
        br(),
        ### MAIN PLOT ###
        conditionalPanel(
          condition = "input.controlPanel == \"results\"",
          dataTableOutput("config_for_this_run"),
          conditionalPanel(
            condition = "input.ExperimentPlotType == \"Performance\"",
            plotOutput("performance_plot")
          ),
          conditionalPanel(
            condition = "input.ExperimentPlotType == \"Statistical properties\"",
            plotOutput("z_scatter_plot")
          ),
          conditionalPanel(
            condition = "input.ExperimentPlotType == \"Associations\"",
            plotOutput("association_plot", width = "100%")
          ),
          conditionalPanel(
            condition = "input.ExperimentPlotType == \"GWAS\"",
            dataTableOutput("gwas_hits_summary"),
            conditionalPanel(
              condition = "input.GWASPlotType == \"Q-Q plot (all z's pooled)\"",
              imageOutput("qqplot")
            ),
            conditionalPanel(
              condition = "input.GWASPlotType == \"Manhattan plot\"",
              imageOutput("manhattan")
            )
          )
        ),
        ### DATATABLE ###
        conditionalPanel(
          condition = "input.controlPanel == \"run_details\"",
          div(DT::dataTableOutput("params_df"), style = "font-size:70%"), br(),
        )
      
      )
      
    ) # end MAIN PANEL
  ) # end FLUID PAGE
) # end SHINY UI