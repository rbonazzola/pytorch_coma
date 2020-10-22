# ShinyUKBB
Code to deploy a Shiny app to visualize the results of the cardiac quantification pipeline on UK Biobank data, and also other non-image-derived data from UKBB participants. 

For more information on Shiny apps, visit [this link](https://shiny.rstudio.com/).

## Requirements
This application has been tested on `R 3.2.3`, and the following packages are required:

- `ggplot2` (3.3.2)
- `dplyr` (1.0.0)
- `shiny` (1.5.0)
- `DT` (0.14)

The versions within parentheses are the ones for which the app was tested, but it is likely to work in newer versions as well.

## Data
Apart from the software packages, input data files must be downloaded into the folder `data/` since these data are protected and cannot be shared herein. For instructions on how to download the data, contact Andres or Rodrigo.

## How to deploy the application on MULTI-X
Request access to the AMI `ShinyApps - R` if you don't have access already.
Launch an instance from this AMI, log into it and execute the following command:

`R -e "shiny::runApp(\"/srv/shiny-server/ShinyUKBB\", port=8080, host=\"0.0.0.0\", launch.browser=F)"`

By deploying the application on the port `0.0.0.0`, it is possible to access it from any IP address without the need of authenticating yourself.
To do this, copy the IP address of the EC2 instance into the address bar of your web browser, and add the suffix `:8080`.  

## To do

- Improve graphical interface by using `Shiny`'s tabsets.
- Use `three` JavaScript library (see [here](https://threejs.org/)) to plot 3D cardiac meshes.
- Add patient diagnoses.
- Add a download button for plots and tables.

If you think of new features to be added to the app, please open an issue describing it or contribute to the code by making a pull request.
