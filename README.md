<h1>
  <p align="center">A Predictive Model to Forecast Victorian COVID-19 Cases</p>
</h1>

<p align="center"><b>Author</b></p>
<a href="https://github.com/j-b-ferguson"><p align="center">Justin Ferguson GitHub</p></a>
<a href="https://www.linkedin.com/in/jf2749/"><p align="center">Justin Ferguson LinkedIn</p></a>
<a href="mailto:justin.benjamin.ferguson@gmail.com?subject=GitHub%20Enquiry"><p align="center">Contact</p></a>

# Introduction

In this notebook, a predictive model using OLS (Ordinary Least Squares)
multiple linear regression is created to forecast next day COVID-19
cases in the state of Victoria. Such considerations are of great
importance in the current health climate. For example, a model to
predict next day cases might be important in enabling the timely
allocation of resources to hospitals for immediate patient treatment.

The linear regression model to be considered requires independent
variables to model a dependent variable - known as the predictors and
response in the machine learning community. In machine learning, a data
set of predictor variables are trained to create a response. This
variable models a set of true values, and the error between them is a
measure of the model's accuracy. Therefore, the challenge in predictive
modelling is to find a set of predictor variables that can train a model
to suitably minimise the error between the response and true values.
Such a model can then be cautiously used to make predictions about the
future.

# Packages


```{.r .fold-show}
library(readr) # Read csv files
library(ggplot2) # Data visualisation
library(dplyr) # Data frame manipulation
library(cowplot) # For ggplot2 grid formats
library(mlr) # For machine learning regression tasks, model training, predictions and performance metrics
library(forecast) # For Boxcox and inverse Boxcox transformations
library(ggResidpanel) # For ggplot2 visualisation of residuals to check OLS linear regression assumptions
library(stringr) # For string manipulations
library(car) # Load statistical tests
library(DT) # For interactive tables
library(tibble) # For tibble manipulation
library(knitr) # For kable tables
```

# Exploratory Data Analysis

## Read Data 

The following model makes use of the Australian COVID-19 data I
previously cleaned in [this
repository](https://github.com/j-b-ferguson/covid-19-australia-preprocessing).


```r
# Read csv files from my github repository
url <- 'https://raw.githubusercontent.com/j-b-ferguson/covid-19-australia-preprocessing/main/Cleaned%20Data%20after%20Preprocessing/covid19_Australia_data_cleaned.csv'
df <- read_csv(url)
kable(df, caption = 'Table 1: The cleaned COVID-19 data imported from my github repository.')
```



Table 1: First 5 results of the cleaned COVID-19 data imported from my github repository.

|Province/State |Date       | Cumulative Cases| Cumulative Tests| Daily Cases| Daily Tests| 7 day daily case moving average| 14 day daily case moving average| 7 day daily test moving average| 14 day daily test moving average|
|:--------------|:----------|----------------:|----------------:|-----------:|-----------:|-------------------------------:|--------------------------------:|-------------------------------:|--------------------------------:|
|NSW            |2020-03-09 |               48|             8371|          10|         363|                               0|                                0|                               0|                                0|
|NSW            |2020-03-10 |               55|            10165|           7|        1794|                               0|                                0|                               0|                                0|
|NSW            |2020-03-11 |               65|            10221|          10|          56|                               0|                                0|                               0|                                0|
|NSW            |2020-03-12 |               92|            14856|          27|        4635|                               0|                                0|                               0|                                0|
|NSW            |2020-03-13 |              119|            16685|          27|        1829|                               0|                                0|                               0|                                0|

## Pre-processing

In creating this model I am interested in predicting the next day
COVID-19 case numbers in the Australian state of Victoria. To suitably
test a predictive model, a truth column `NextDayCases` is created from
the given data set. This column is required in order to measure the
error created by OLS linear regression between the model response and
true values. The truth values will consist of the values in the
`DailyCases` column that lead in time by one day. More concisely,
`NextDayCases[i]` = `df.vic$DailyCases[i+1]`, where `i` is any given day
in the set of allowed days. Overall, this simple model considers if
predictors like `DailyCases`, `SvnDayCaseAvg`, etc., can create a model
response that accurately predicts those values in the `NextDayCases`
column.


```r
# Make a copy of the original data frame for graph axis labelling
df.new <- df

# Rename to simple column names
colnames(df.new) <- c('State', 'Date', 
                      'CumulativeCases', 'CumulativeTests', 
                      'DailyCases', 'DailyTests', 
                      'SvnDayCaseAvg', 'FrtnDayCaseAvg', 
                      'SvnDayTestAvg', 'FrtnDayTestAvg')                  

# Convert data type
df.new$State <- df.new$State %>% as.factor()

# Consider just Victorian COVID-19 cases
df.vic <- df.new %>% filter(State == 'VIC') %>% select(-State)

# Filter out observations with zero values
df.vic <- df.vic %>% filter(DailyCases != 0, SvnDayCaseAvg != 0, FrtnDayCaseAvg != 0)

# Create new column and fill with missing values
df.vic$NextDayCases <- NA

# Define values in NextDayCases column as DailyCases[i+1] (the day after)
for (i in 1:nrow(df.vic)) {
  if (i < nrow(df.vic)) {
    df.vic$NextDayCases[i] <- df.vic$DailyCases[i+1]
  } else {
    df.vic$NextDayCases[i] <- NA
  }
}

# Keep only complete observations
df.vic <- df.vic[complete.cases(df.vic),]

# Now remove the date column as it is an invalid type for the regression task
df.vic <- df.vic %>% select(-Date)

# Show modified table
kable(df.vic, caption = 'Table 2: COVID-19 table showing only Victorian data, with next day case values added.')
```



Table 2: First 5 results of the COVID-19 table showing only Victorian data, with next day case values added.

| CumulativeCases| CumulativeTests| DailyCases| DailyTests| SvnDayCaseAvg| FrtnDayCaseAvg| SvnDayTestAvg| FrtnDayTestAvg| NextDayCases|
|---------------:|---------------:|----------:|----------:|-------------:|--------------:|-------------:|--------------:|------------:|
|             876|           30300|         52|       3000|            67|             58|          2614|           2164|           96|
|             972|           33300|         96|       3000|            72|             63|          2857|           2200|           51|
|            1023|           35300|         51|       2000|            72|             64|          3071|           2271|           68|
|            1091|           37300|         68|       2000|            74|             69|          3157|           2273|           49|
|            1140|           39300|         49|       2000|            73|             73|          2857|           2262|           30|

## Visualisation

Before creating the machine learning task, it is helpful to first
visualise the bivariate relationships between the `NextDayCases`
variable and the potential model predictors to identify simple linear
relationships.

Below, the `SvnDayCaseAvg` and `FrtnDayCaseAvg` variables show linearity
with respect to the response variable, and the `DailyCases` variable
appears mostly linear but strongly affected by the influence of outliers
at high values. Another observation is the increasing error between the
data points and the regression lines in the forward direction - this is
an indication of unequal variance (heteroscedasticity).
Heteroscedasticity is undesired in a linear regression model and would
indicate that some kind of transformation is required.


```r
# Save column names as vectors
Y <- df.vic[9] %>% unlist() %>% as.vector()
colNames <- names(df.vic)[1:8]

# A loop to create plots showing the predictor variables plotted against "NextDayCases"
for (i in colNames) {
  plt <- ggplot(data = df.vic, aes_string(x=i, y=Y)) +
                geom_point(colour='black') +
                geom_smooth(method='loess', se=FALSE, colour='red') +
                xlab(str_to_title(names(df[match(i, colNames)+2]))) +
                theme(axis.title.y = element_blank())
  assign(paste0('gg.', i), plt)
}

# Create grid of plots
plt.grid <- plot_grid(gg.CumulativeCases, gg.CumulativeTests,
                      gg.DailyCases, gg.DailyTests,
                      gg.FrtnDayCaseAvg, gg.FrtnDayTestAvg,
                      gg.SvnDayCaseAvg, gg.SvnDayTestAvg, 
                      nrow = 4, 
                      ncol = 2)

# Plot title
title <- ggdraw() + 
            draw_label("The Bivariate Relationships between the Next Day Cases Variable and the Potential Model Predictors", 
                       fontface='bold')

# Show plot grid
plt.grid.title <- plot_grid(title, plt.grid, ncol=1, rel_heights=c(0.1, 1))
plt.grid.title
```

![Figure 1: The bivariate relationships between the "NextDayCases" Variable and the potential model predictors.](/plots/plt.grid.title.svg)

# Model Assumptions

The OLS linear regression model assumptions must now be defined before
continuing further.

<p class="border-p">
__*1. Linearity:*__ There is a linear relationship between the
independent variable, x, and the dependent variable, y. 
<br><br> 
__*2.Independence:*__ Variables are independent of each other as to not be
derived from the same source. Or for time series data, correlation
between time-shifted copies of variables is not present - known as
autocorrelation. 
<br><br> 
__*3. Homoscedasticity:*__ The residuals haveconstant variance for any 
value of x. 
<br><br> 
__*4. Normality:*__ The residuals are normally distributed.
</p>

On the basis of assumption 2, a simple linear regression model from the
set of bivariate relationships above should be ignored because
`NextDayCases` is a time-shifted copy of the `DailyCases` variable.
Therefore, autocorrelation should exist in the relationship between
`NextDayCases` and `DailyCases`.

This is confirmed with the Durbin-Watson hypothesis test for
autocorrelation from the `car` package.


<p class="border-p">
__*<i>H<sub>0</sub></i> (null hypothesis):*__ There is no correlation among residuals.
<br>
__*<i>H<sub>A</sub></i> (alternative hypothesis):*__ There is autocorrelation.
</p>


```{.r .fold-show}
# Create a linear regression model and run the Durbin-Watson test
simplemodel <- lm(NextDayCases ~ DailyCases, data = df.vic)
durbinWatsonTest(simplemodel)
```

```
##  lag Autocorrelation D-W Statistic p-value
##    1      -0.3803768      2.758981       0
##  Alternative hypothesis: rho != 0
```

At the significance level &alpha; = 0.05, <i>p</i> < 0.05 in the
Durbin-Watson test; and so the null hypothesis is rejected, thus
confirming a statistically significant autocorrelation in this simple
linear model. Further, the corresponding bivariate relationships
including `SvnDayCaseAvg` and `FrtnDayCaseAvg` are sourced from values
of `DailyCases` and can also be assumed to violate independence.

Autocorrelation can be prevented by creating a multiple linear
regression model. For example, observe the insignificant result obtained
from the Durbin-Watson test for the following set of predictors.


```{.r .fold-show}
# Create a multiple linear regression model and run the Durbin-Watson test
multiplemodel <- lm(NextDayCases ~ DailyCases + SvnDayCaseAvg, data = df.vic)
durbinWatsonTest(multiplemodel)
```

```
##  lag Autocorrelation D-W Statistic p-value
##    1      0.03035635      1.937744   0.584
##  Alternative hypothesis: rho != 0
```
# Supervised Machine Learning 1

## Dimension Reduction 1

We established in the last section that the linear regression model
requires at least two predictors in order to satisfy assumption 2. A
regression task using the `mlr` package is now created in order to find
a set of suitable predictors for the regression model target. Creating
the regression task returns the number of observations and features used
in the model. The number of features initially returns the total number
of variables in the `df.vic` data frame - 8 at this point.


```{.r .fold-show}
# Create a regression task
cases.task <- makeRegrTask(data = df.vic, target = "NextDayCases")
cases.task
```

```
## Supervised task: df.vic
## Type: regr
## Target: NextDayCases
## Observations: 188
## Features:
##    numerics     factors     ordered functionals 
##           8           0           0           0 
## Missings: FALSE
## Has weights: FALSE
## Has blocking: FALSE
## Has coordinates: FALSE
```

Not all features in a regression model task will help to generalise
towards model accuracy. In fact, it is common to minimise the number of
features in order to avoid overfitting, as well as prevent data sparsity
by association with the "curse of dimensionality". Both issues are detrimental
to model accuracy.

Dimension reduction with feature selection is a technique used to
minimise the number of features before training a predictive model. In
this multiple linear regression model, a sensible method is to keep
features with a considerably high Pearson correlation
coefficient; and so removing features with weak linear relationships, up to
an arbitrarily chosen value of 0.7.


```{.r .fold-show}
# Data set dimension reduction using feature selection
cases.task <- filterFeatures(task = cases.task, method = 'linear.correlation', threshold = 0.7)
getTaskFeatureNames(task = cases.task)
```

```
## [1] "DailyCases"     "SvnDayCaseAvg"  "FrtnDayCaseAvg" "SvnDayTestAvg"  "FrtnDayTestAvg"
```

## Checking the Residuals

A linear model is now created with the minimised feature set; and this
model must satisfy the linear regression assumptions mentioned earlier.
The `ggResidpanel` package is used below to create a grid of diagnostic
plots to check these assumptions with respect to the model.


```r
# Create the multiple linear regression model
linearmodel <- lm(NextDayCases ~ DailyCases + SvnDayCaseAvg + FrtnDayCaseAvg + SvnDayTestAvg + FrtnDayTestAvg, data = df.vic)

# Check OLS assumptions by inspecting the residual plots
residuals <- resid_panel(linearmodel, plots = "R", smoother = TRUE, qqbands = TRUE, theme = "gray")
residuals
```

![Figure 2: Diagnostic plots to test the linear regression model assumptions.](/plots/residpanel.svg)

__*Linearity:*__ The model *does not* satisfy the requirement of
linearity as the LOESS lines in the residual and location-scale plots
are not horizontal.

__*Independence:*__ The requirement for independence has been satisfied
as the model now contains multiple predictors and does not
autocorrelate.

__*Homoscedasticity:*__ The non-constant variance observed in the
residual and location-scale plots shows the requirement of
homoscedasticity has been violated.

__*Normality:*__ The requirement of normality of residuals has been
violated as there are many data points that diverge outside of the 95%
confidence interval bands in the Q-Q plot.

Therefore, this linear model cannot be used to make predictions about
next day COVID-19 cases. All is not lost, however; as a Boxcox
transformation applied to the independent and dependent variables will
normalise their values and perhaps improve this linear model.

Before continuing onward, let us first address the residual-leverage
plot. Data points with a relatively large Cook's distance have
significant influence on the model; and might therefore be identified as
a potential outlier. In this plot, we can identify an outlier with a
Cook's distance greater than 1. This corresponds to the following
observation; and has been removed from the data set to improve model
accuracy going forward.


```{.r .fold-show}
# Show and remove observations with a Cook's distance greater than 1 in the linear model 
linearmodel$model[cooks.distance(linearmodel) > 1,]
```

```
##     NextDayCases DailyCases SvnDayCaseAvg FrtnDayCaseAvg SvnDayTestAvg FrtnDayTestAvg
## 122          579        694           410            375         24106          25522
```

```{.r .fold-show}
df.vic <- df.vic[-122,]
```

# Transforming the Model

## Boxcox Transformation
A Boxcox transformation is now applied to the original table of
variables to improve normality and thus create a model that conforms
with respect to the required assumptions. The table below shows the
transformed values; and the diagnostic plots reveal that all assumptions
have now been satisfied; specifically, the residual and location plots
now satisfy linearity, the residual plot is homoscedastic, and the data
points in the Q-Q plot are normally distributed.


```{.r .fold-show}
# Boxcox transformation the original data frame to improve the model
df.vicx <- bind_cols(NextDayCasesx = BoxCox(df.vic$NextDayCases, lambda = BoxCox.lambda(df.vic$NextDayCases, method = 'guerrero')),
                     DailyCasesx = BoxCox(df.vic$DailyCases, lambda = BoxCox.lambda(df.vic$DailyCases, method = 'guerrero')), 
                     SvnDayCaseAvgx = BoxCox(df.vic$SvnDayCaseAvg, lambda = BoxCox.lambda(df.vic$SvnDayCaseAvg, method = 'guerrero')),
                     FrtnDayCaseAvgx = BoxCox(df.vic$FrtnDayCaseAvg, lambda = BoxCox.lambda(df.vic$FrtnDayCaseAvg, method = 'guerrero')),
                     SvnDayTestAvgx = BoxCox(df.vic$SvnDayTestAvg, lambda = BoxCox.lambda(df.vic$SvnDayTestAvg, method = 'guerrero')),
                     FrtnDayTestAvgx = BoxCox(df.vic$FrtnDayTestAvg, lambda = BoxCox.lambda(df.vic$FrtnDayTestAvg, method = 'guerrero')))

kable(round(df.vicx, 3), caption = 'Table 3: Victoria COVID-19 Boxcox transformed data.')
```



Table 3: First 5 results of the Victoria COVID-19 Boxcox transformed data.

| NextDayCasesx| DailyCasesx| SvnDayCaseAvgx| FrtnDayCaseAvgx| SvnDayTestAvgx| FrtnDayTestAvgx|
|-------------:|-----------:|--------------:|---------------:|--------------:|---------------:|
|         7.068|       6.210|          7.822|           6.737|          7.439|           5.854|
|         5.712|       7.735|          8.048|           6.951|          7.518|           5.863|
|         6.310|       6.165|          8.048|           6.992|          7.582|           5.881|
|         5.631|       6.853|          8.135|           7.191|          7.607|           5.881|
|         4.687|       6.072|          8.092|           7.342|          7.518|           5.879|


```r
# Create a new linear model with the Boxcox transformed values and then check the OLS linear regression assumptions again
linearmodelx <- lm(NextDayCasesx ~ DailyCasesx + SvnDayCaseAvgx + FrtnDayCaseAvgx + SvnDayTestAvgx + FrtnDayTestAvgx, data = df.vicx)
resid_panel(linearmodelx, "R", smoother = TRUE, qqbands = TRUE, theme = "gray")
```

![Figure 3: Diagnostic plots to test the transformed linear regression model assumptions.](/plots/residpanelX.svg)

## Dimension Reduction 2

With the linear regression assumptions all satisfied, a more advanced form of feature selection can now be 
used on the transformed model. Backwards elimination is a technique used to sequentially remove 
predictors of <i>p</i> > 0.05; these predictors are less likely to contribute meaningfully 
to the linear model. The result is a model with improved accuracy and performance; as well as a reduction in 
overfitting.

<p class="border-p">
__*Starting Features:*__ ```DailyCasesx```, ```SvnDayCaseAvgx```, ```FrtnDayCaseAvgx```, ```SvnDayTestAvgx```, ```FrtnDayTestAvgx```
</p>


```r
# Create first linear model with all the transformed predictors and target variables
elim1 <- lm(NextDayCasesx ~ DailyCasesx + SvnDayCaseAvgx + FrtnDayCaseAvgx + SvnDayTestAvgx + FrtnDayTestAvgx, data = df.vicx) %>% summary()

# Extract summary information including p values
elim1.summary <- elim1$coefficients %>% round(2) %>% as.data.frame() %>% rownames_to_column()
elim1.summary <- elim1.summary[, c(1,5)]

# Rename and format table
col.names <- c('Predictor', 'Pr (> |t|)')
colnames(elim1.summary) <- col.names
elim1.summary %>% kable(caption = 'Table 4: Backward elimination step one.')
```



Table 4: Backward elimination step one.

|Predictor       | Pr (> &#124;t&#124;)|
|:---------------|--------------------:|
|(Intercept)     |                 0.80|
|DailyCasesx     |                 0.10|
|SvnDayCaseAvgx  |                 0.00|
|FrtnDayCaseAvgx |                 0.00|
|SvnDayTestAvgx  |                 0.37|
|FrtnDayTestAvgx |                 0.65|


__*Predictor selected for elimination:*__ ```FrtnDayTestAvgx```



```r
# Create second linear model with one predictor removed
elim2 <- lm(NextDayCasesx ~ DailyCasesx + SvnDayCaseAvgx + FrtnDayCaseAvgx + SvnDayTestAvgx, data = df.vicx) %>% summary()

# Extract summary information including p values
elim2.summary <- elim2$coefficients %>% round(2) %>% as.data.frame() %>% rownames_to_column()
elim2.summary <- elim2.summary[, c(1,5)]

# Rename and format table
colnames(elim2.summary) <- col.names
elim2.summary %>% kable(caption = 'Table 5: Backward elimination step two.')
```



Table 5: Backward elimination step two.

|Predictor       | Pr (> &#124;t&#124;)|
|:---------------|--------------------:|
|(Intercept)     |                 0.38|
|DailyCasesx     |                 0.09|
|SvnDayCaseAvgx  |                 0.00|
|FrtnDayCaseAvgx |                 0.00|
|SvnDayTestAvgx  |                 0.08|


__*Predictor selected for elimination:*__ ```DailyCasesx```



```r
# Create third linear model with one predictor removed
elim3 <- lm(NextDayCasesx ~ SvnDayCaseAvgx + FrtnDayCaseAvgx + SvnDayTestAvgx, data = df.vicx) %>% summary()

# Extract summary information including p values
elim3.summary <- elim3$coefficients %>% round(2) %>% as.data.frame() %>% rownames_to_column()
elim3.summary <- elim3.summary[, c(1,5)]

# Rename and format table
colnames(elim3.summary) <- col.names
elim3.summary %>% kable(caption = 'Table 6: Backward elimination step three.')
```



Table 6: Backward elimination step three.

|Predictor       | Pr (> &#124;t&#124;)|
|:---------------|--------------------:|
|(Intercept)     |                 0.34|
|SvnDayCaseAvgx  |                 0.00|
|FrtnDayCaseAvgx |                 0.00|
|SvnDayTestAvgx  |                 0.05|


__*Optimal features:*__ ```SvnDayCaseAvgx```, ```FrtnDayCaseAvgx```, ```SvnDayTestAvgx```


So, we can now modify the ```df.vicx``` table to include only the optimal features in 
model fitment.


```{.r .fold-show}
# Modify df.vicx data frame as per backward elimination
df.vicx.optimal <- df.vicx[, c('NextDayCasesx', 'SvnDayCaseAvgx', 'FrtnDayCaseAvgx', 'SvnDayTestAvgx')]
```

# Supervised Machine Learning 2

A generalised linear predictive model can now be trained and tested with
a supervised machine learning task from the `mlr` package. The process
is recorded within the subsections below.

## Regression Task and Learner

A new regression task is first created with the transformed and optimised 
predictor and target variables. The features and observations of the task are less than
for the original ```case.task``` due to feature selection and outlier removal.


```{.r .fold-show}
# Update the regression task with the transformed values from the df.vicx.optimal data frame
cases.taskx <- makeRegrTask(data = df.vicx.optimal, target = 'NextDayCasesx')
cases.taskx 
```

```
## Supervised task: df.vicx.optimal
## Type: regr
## Target: NextDayCasesx
## Observations: 187
## Features:
##    numerics     factors     ordered functionals 
##           3           0           0           0 
## Missings: FALSE
## Has weights: FALSE
## Has blocking: FALSE
## Has coordinates: FALSE
```

A generalised linear model algorithm is now appropriately chosen to learn 
from the task and a random subset of the data set for linear regression. 


```{.r .fold-show}
# Create a generalised linear regression learner
lrn <- makeLearner(cl = 'regr.glm')
lrn
```

```
## Learner regr.glm from package stats
## Type: regr
## Name: Generalized Linear Regression; Short name: glm
## Class: regr.glm
## Properties: numerics,factors,se,weights
## Predict-Type: response
## Hyperparameters: family=gaussian,model=FALSE
```

## Fit Model and Evaluate Performance

The learner and task are now used in combination with a leave-one-out (LOO)
cross validation (CV) resampling method to fit the model.

Cross validation is a resampling method which randomly splits the data
into <i>k</i> number of subsets (or 'folds'). The model is then fitted 
<i>k</i> times, which exactly corresponds to the number of folds used as 
training data sets.

When cross validation is used exhaustively, all possible combinations of 
resampling are explored; this corresponds to the total number of 
observations in the data set. This is otherwise known as LOO CV.

> *Why use LOO CV in the first place?* 
>
>Randomly splitting the data into training and test data subsets just once is a commonly used approach to fit 
a model. However, this approach is not a reliable approach for testing as the model performance can vary between different test sets. 
LOO CV is a reliable method to measure the model performance because each training/test split is explored and averaged overall. 

Model performance is measured in terms of a loss function; a metric to 
evaluate the average error between the predicted response and true values.
Selecting an appropriate loss function requires an understanding of the domain
of the machine learning problem; this includes any real-world constraints or 
conditions.

If the domain of the problem requires large errors (outliers) to carry more 
weight than smaller errors, then a suitable loss function is mean-square error (MSE).
Conversely, if the domain of the problem requires equal weighting for all errors,
then mean absolute error (MAE) should be used.

For this linear regression model, error weighting for the transformed COVID-19 cases 
should scale linearly; and so, MAE is appropriately chosen.

Running the resampling method below to fit the model achieves an average model performance of 
0.7102347.


```{.r .fold-show}
# Set seed value 
set.seed(1234)

# Set resample method as leave-one-out cross validation
rdesc <- makeResampleDesc(method = "LOO")

# Perform resampling of the learner on the task with leave-one-out cross validation.
model.performance <- resample(learner = lrn, task = cases.taskx, resampling = rdesc, measures = list(mae))
model.performance
```

```
## Resample Result
## Task: df.vicx.optimal
## Learner: regr.glm
## Aggr perf: mae.test.mean=0.7102347
## Runtime: 0.844511
```

# Model Limitations

Most predictive models have limitations; this is no exception. To investigate possible constraints, a functional predictive model must first be created from a train/test data split. Make note that cross-validation was performed for testing purposes only; and a working model need only require a conventional data split.


```{.r .fold-show}
# Split the data into random subsets
set.seed(1234)
n <- nrow(df.vicx.optimal)
train.set <- sample(n, size = 4/5*n)
test.set <- df.vicx.optimal[setdiff(1:n, train.set),]

# Train the learner on the training data set
model.optimal <- train(lrn, cases.taskx, train.set)
```

In the table below are the Boxcox transformed predictions, 95% upper and lower prediction bounds, and true values of the fitted model.


```r
# Get model prediction and true values
prediction.points <- predict(model.optimal, newdata = test.set) %>% 
  as.data.frame() %>% 
  round(3)

# Get 95% prediction interval values
prediction.interval <- predict.lm(
  lm(NextDayCasesx ~ SvnDayCaseAvgx + FrtnDayCaseAvgx + SvnDayTestAvgx, data = df.vicx.optimal), 
  newdata = test.set, 
  interval = "predict") %>% 
  as.data.frame() %>% 
  round(3)

# Combine data into one data frame
predictions<- bind_cols(prediction.points, prediction.interval)
predictions <- predictions[-2]

# Edit names
colnames(predictions) <- c('True Values', 'Predictions', '95% Lower Bound', '95% Upper Bound')

# Output table
kable(predictions, caption = 'Table 7: Table of transformed predictions, true values, and 95% upper and lower prediction bounds.')
```



Table 7: First 5 results of the table of transformed predictions, true values, and 95% upper and lower prediction bounds.

| True Values| Predictions| 95% Lower Bound| 95% Upper Bound|
|-----------:|-----------:|---------------:|---------------:|
|       7.068|       5.749|           3.915|           7.582|
|       4.687|       5.724|           3.898|           7.549|
|       4.284|       2.575|           0.734|           4.417|
|       1.214|       2.704|           0.874|           4.535|
|       1.214|       2.127|           0.305|           3.949|


And below is the mapping between the true values and the Boxcox transformed true values of ```NextDayCases``` from the ```df.vic``` and ```df.vicx.optimal``` tables.


```{.r .fold-hide}
# Merge together the original and transformed next day case values into a new data frame
nextdaycases.compare <- bind_cols(df.vic$NextDayCases ,df.vicx.optimal$NextDayCasesx) %>% round(3)

# Tidy and output merged table
colnames(nextdaycases.compare) <- c('True Values', 'Boxcox Transformed')
kable(nextdaycases.compare, caption = 'Table 8: Mapping between Boxcox transformed values and true values of next day cases.')
```



Table 8: First 5 results of the mapping between Boxcox transformed values and true values of next day cases.

| True Values| Boxcox Transformed|
|-----------:|------------------:|
|          96|              7.068|
|          51|              5.712|
|          68|              6.310|
|          49|              5.631|
|          30|              4.687|

In order to extract meaningful next day COVID-19 case predictions from the model, any prediction must undergo an inverse Boxcox transformation to map its predicted value to its true value. The values inside tables 7 and 8 are visualised underneath and reveal this mapping process rises exponentially. Therefore, for increasingly large Boxcox predictions, the 95% prediction interval will also grow exponentially, and so affect model accuracy.


```r
# Create linear regression plot with the 95% prediction interval
plot1 <- ggplot(data = predictions, mapping = aes(y = `True Values`)) +
            geom_point(mapping = aes(x = `Predictions`)) +
            geom_smooth(mapping = aes(x = `Predictions`), method = "lm", se = FALSE, color = 'red') +
            geom_ribbon(mapping = aes(x = `Predictions`, ymin = `95% Lower Bound`, ymax = `95% Upper Bound`), fill = 'steelblue', alpha = 0.2)

# Create a plot to compare true predicted values against mapped predicted Boxcox values
plot2 <- ggplot(data = nextdaycases.compare, mapping = aes(x = `Boxcox Transformed`, y = `True Values`)) +
            geom_line(color = 'red', size = 1) +
            geom_point()

# Combine plots as grid         
plt <- plot_grid(plot1, plot2)

# Create a title
title <- ggdraw() + draw_label("The Transformed Linear Regression Model and Mapping between Boxcox Transformed Values and True Values", fontface='bold')

# Output grid plot
plot_grid(title, plt, ncol=1, rel_heights=c(0.1, 1))
```

![Figure 4: The transformed linear regression model and mapping between Boxcox transformed values and true values.](/plots/regression_mapping.svg)

A further analysis of this limitation requires the average 95% prediction width of the model; this is given in the output below. Table 9 below shows uses arbitrary Boxcox prediction values - ranging from 1 to 10 - and the average prediction width to formulate Boxcox prediction intervals. These values and intervals are mapped to into the true values, thus revealing the exponentially increasing true prediction width.


```r
# Calculate and show prediction interval width
prediction.width <- ((sum(predictions$`95% Upper Bound`) - sum(predictions$`95% Lower Bound`)) / nrow(predictions)) %>% round(3)
prediction.width
```

```
## [1] 3.625
```


```r
# Code to produce a table showing 'Boxcox Values', 'Boxcox Prediction Intervals', 'True Values', 'True Prediction Intervals' and 'True Prediction Width'.
width <- prediction.width / 2
boxcox.values <- seq(1, 10, by = 1)
true.values <- InvBoxCox(boxcox.values, lambda = BoxCox.lambda(df.vic$NextDayCases, method = 'guerrero')) %>% round(0)
true.intervals <- c()
boxcox.intervals <- c()
lower.bound <- c()
upper.bound <- c()
true.width <- c()

for (i in 1:10) {
  boxcox.intervals[i] <- paste0('[', as.character(i - width), ', ', as.character(i + width), ')')
  lower.bound[i] <- InvBoxCox(i - width, lambda = BoxCox.lambda(df.vic$NextDayCases, method = 'guerrero')) %>% round(0)
  upper.bound[i] <- InvBoxCox(i + width, lambda = BoxCox.lambda(df.vic$NextDayCases, method = 'guerrero')) %>% round(0)
  true.intervals[i] <- paste0('[', as.character(lower.bound[i]), ', ', as.character(upper.bound[i]), ')')
  true.width[i] <- upper.bound[i] - lower.bound[i]
}

model.error <- data.frame(boxcox.values, boxcox.intervals, true.values, true.intervals, true.width)
colnames(model.error) <- c('Boxcox Values', 'Boxcox Prediction Intervals', 'True Values', 'True Prediction Intervals', 'True Prediction Width')

kable(model.error, caption = 'Table 9: A table of prediction values and intervals to highlight the increasing width of the true prediction interval.')
```



Table 9: A table of prediction values and intervals to highlight the increasing width of the true prediction interval.

| Boxcox Values|Boxcox Prediction Intervals | True Values|True Prediction Intervals | True Prediction Width|
|-------------:|:---------------------------|-----------:|:-------------------------|---------------------:|
|             1|[-0.8125, 2.8125)           |           3|[0, 10)                   |                    10|
|             2|[0.1875, 3.8125)            |           6|[1, 18)                   |                    17|
|             3|[1.1875, 4.8125)            |          11|[3, 32)                   |                    29|
|             4|[2.1875, 5.8125)            |          20|[6, 54)                   |                    48|
|             5|[3.1875, 6.8125)            |          35|[12, 86)                  |                    74|
|             6|[4.1875, 7.8125)            |          59|[23, 132)                 |                   109|
|             7|[5.1875, 8.8125)            |          93|[39, 197)                 |                   158|
|             8|[6.1875, 9.8125)            |         143|[64, 287)                 |                   223|
|             9|[7.1875, 10.8125)           |         212|[101, 408)                |                   307|
|            10|[8.1875, 11.8125)           |         307|[154, 567)                |                   413|

# Conclusion

A multiple linear regression model has been created from 3 Boxcox transformed predictors to predict next day COVID-19 cases with Victorian data. Running leave-one-out cross validation achieved an average model performance accuracy of MAE = 0.7102347. To obtain meaningful data, the predicted values must be mapped into meaningful values with an inverse Boxcox transformation. This process reveals an increasingly large 95% true prediction interval, assumingly due to the heteroscedasticity of original data. Therefore, large predictions should be cautiously interpreted due to reducing model accuracy.
