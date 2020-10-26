shinyUI(
  
  fluidPage(
    
    includeHTML("www/coma.html"),  
    # titlePanel("CoMA Results Explorer"),
    
    sidebarPanel(
      tabsetPanel(
        id = "controlPanel",
        tabPanel(title = "Summaries", uiOutput("SummTab"), value = "summaries"),
        tabPanel(title = "Latent space", uiOutput("Experiment"), value = "experiment_details"),
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
          condition = "input.controlPanel == \"gwas\"",
          imageOutput("qqplot_pooled"), br(),
        ),
        conditionalPanel(
          condition = "input.controlPanel == \"experiment_details\" || input.controlPanel == \"summaries\"",
          plotOutput("plot", brush = brushOpts(id = "plot1_brush"))
        ),
        
        ### DATATABLE ###
        conditionalPanel(
          condition = "input.controlPanel == \"run_details\"",
          div(DT::dataTableOutput("params_df"), style = "font-size:70%"), br(),
        ),
        conditionalPanel(
          condition = "input.controlPanel == \"latent_space\"", # || input.controlPanel == \"assoc\"",
          DT::dataTableOutput("brush_info_z"), br(),
        ),
      )
    )
    
))

