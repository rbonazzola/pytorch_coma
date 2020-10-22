shinyUI(
  
  fluidPage(
    
    titlePanel("CoMA Results Explorer"),
    
    sidebarPanel(
      tabsetPanel(
        id = "controlPanel",
        tabPanel(title = "Summaries", uiOutput("SummTab"), value = "summaries"),
        tabPanel(title = "Model performance", uiOutput("PerfTab"), value = "performance"),
        tabPanel(title = "Latent space", uiOutput("ZTab"), value = "latent_space"),
        tabPanel(title = "Associations", uiOutput("AssocTab"), value = "assoc"),
        tabPanel(title = "GWAS", uiOutput("GWASTab"), value = "gwas"),
        tabPanel(title = "Docs", htmlOutput("docsTab"), value = "docs")
      )
    ),
  
    mainPanel(
      fluidRow(
        br(),
        plotOutput("plot", brush = brushOpts(id = "plot1_brush")),
        conditionalPanel(
          condition = "input.controlPanel == \"summaries\"",
          div(DT::dataTableOutput("params_df"), style = "font-size:70%"), br(),
        ),
        conditionalPanel(
          condition = "input.controlPanel == \"latent_space\"", # || input.controlPanel == \"assoc\"",
          DT::dataTableOutput("brush_info_z"), br(),
        ),
        conditionalPanel(
          condition = "input.controlPanel == \"gwas\"",
          imageOutput("qqplot_pooled"), br(),
        )
      )
    )
    
  )
)

# downloadButton("plotFile", "Download plot"),
# downloadButton("dataFile", "Download data as dataframe"),
# uiOutput("loading"),
# if (input$controlPanel == "X vs. Y") {
#   DT::dataTableOutput("brush_info")
# }