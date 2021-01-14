<h1>
  <p align="center">A Predictive Model to Forecast Australian COVID-19 Cases</p>
</h1>

<p align="center"><b>Author</b></p>
<a href="https://github.com/j-b-ferguson"><p align="center">Justin Ferguson GitHub</p></a>
<a href="https://www.linkedin.com/in/jf2749/"><p align="center">Justin Ferguson LinkedIn</p></a>
<a href="mailto:justin.benjamin.ferguson@gmail.com?subject=GitHub%20Enquiry"><p align="center">Contact</p></a>

<h2><p align=center>Executive Summary</h2>
In this notebook a predictive model using ordinary least squares linear regression is created to forecast next day COVID-19 cases in the state of Victoria. 

<h2><p align=center>Packages</p></h2>

```r
library(readr)
library(magrittr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(GGally)
library(mlr)
library(forecast)
library(ggResidpanel)
```

<h2>Read Data</h2>

```r
url <- 'https://raw.githubusercontent.com/j-b-ferguson/covid-19-australia-preprocessing/main/Cleaned%20Data%20after%20Preprocessing/covid19_Australia_data_cleaned.csv'
df <- read_csv(url)
```

<h2><p align=center>Preprocessing</p></h2>
