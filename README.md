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

<h2><p align=center>Read Data</p></h2>

```r
url <- 'https://raw.githubusercontent.com/j-b-ferguson/covid-19-australia-preprocessing/main/Cleaned%20Data%20after%20Preprocessing/covid19_Australia_data_cleaned.csv'
df <- read_csv(url)
```

<h2><p align=center>Preprocessing</p></h2>

```r
# Rename to simple column names
colnames(df) <- c('State', 'Date', 
                  'CumulativeCases', 'CumulativeTests', 
                  'DailyCases', 'DailyTests', 
                  'SvnDayCaseAvg', 'FrtnDayCaseAvg', 
                  'SvnDayTestAvg', 'FrtnDayTestAvg')                  
```

```r
# Convert data type
df$State <- df$State %>% as.factor()
```

```r
# Consider just Victorian COVID-19 cases
df.vic <- df %>% filter(State == 'VIC')
```

```r
# Filter out observations with zero values
df.vic <- df.vic %>% filter(DailyCases != 0, SvnDayCaseAvg != 0, FrtnDayCaseAvg != 0)
```

```r
# Create new column and fill with missing values
df.vic$NextDayCases <- NA
```

```r
# Define values in NextDayCases column as DailyCases[i+1] (the day after)
for (i in 1:nrow(df.vic)) {
  if (i < nrow(df.vic)) {
    df.vic$NextDayCases[i] <- df.vic$DailyCases[i+1]
  } else {
    df.vic$NextDayCases[i] <- NA
  }
}
```

```r
# Keep only complete observations
df.vic <- df.vic[complete.cases(df.vic),]
```

```r
# Now remove the date column as it is an invalid type for the regression task
df.vic <- df.vic %>% select(-Date)
```

```r
# Show data frame
str(df.vic)
```


