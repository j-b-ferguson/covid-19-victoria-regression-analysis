
# Packages ----------------------------------------------------------------

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


# Exploratory Data Analysis -----------------------------------------------

## Read Data ---------------------------------------------------------------

# Read csv files from my github repository
url <- 'https://raw.githubusercontent.com/j-b-ferguson/covid-19-australia-preprocessing/main/Cleaned%20Data%20after%20Preprocessing/covid19_Australia_data_cleaned.csv'
df <- read_csv(url)
datatable(df, options = list(pageLength =  5), caption = 'Table 1: The cleaned COVID-19 data imported from my github repository.')

## Pre-processing ----------------------------------------------------------

# Make a copy of the original data frame for graph axis labelling
df.new <- df

# Rename to simple column names
colnames(df.new) <- c('State', 'Date', 
                      'CumulativeCases', 'CumulativeTests', 
                      'DailyCases', 'DailyTests', 
                      'SvnDayCaseAvg', 'FrtnDayCaseAvg', 
                      'SvnDayTestAvg', 'FrtnDayTestAvg')                  

# Convert data type
df.new$State <- df.new$State %>% 
  as.factor()

# Consider just Victorian COVID-19 cases
df.vic <- df.new %>% filter(State == 'VIC') %>% 
  select(-State)

# Filter out observations with zero values
df.vic <- df.vic %>% 
  filter(DailyCases != 0, SvnDayCaseAvg != 0, FrtnDayCaseAvg != 0)

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
df.vic <- df.vic %>% 
  select(-Date)

# Show modified table
datatable(df.vic, options = list(pageLength =  5), caption = 'Table 2: COVID-19 table showing only Victorian data, with next day case values added.')

## Visualisation -----------------------------------------------------------

# Save column names as vectors
Y <- df.vic[9] %>% 
  unlist() %>% 
  as.vector()

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

# Save plot grid
plt.grid.title <- plot_grid(title, plt.grid, ncol=1, rel_heights=c(0.1, 1))
svglite::svglite('~/COVID-19-ANALYSIS/plots/plt.grid.title.svg')
plt.grid.title
dev.off()




# Model Assumptions -------------------------------------------------------

# Create a linear regression model and run the Durbin-Watson test
simplemodel <- lm(NextDayCases ~ DailyCases, data = df.vic)
durbinWatsonTest(simplemodel)

# Create a multiple linear regression model and run the Durbin-Watson test
multiplemodel <- lm(NextDayCases ~ DailyCases + SvnDayCaseAvg, data = df.vic)
durbinWatsonTest(multiplemodel)



# Supervised Machine Learning 1 -------------------------------------------

## Dimension Reduction 1 ---------------------------------------------------

# Create a regression task
cases.task <- makeRegrTask(data = df.vic, target = "NextDayCases")
cases.task

# Data set dimension reduction using feature selection
cases.task <- filterFeatures(task = cases.task, method = 'linear.correlation', threshold = 0.7)
getTaskFeatureNames(task = cases.task)

## Checking the Residuals --------------------------------------------------

# Create the multiple linear regression model
linearmodel <- lm(NextDayCases ~ DailyCases + SvnDayCaseAvg + FrtnDayCaseAvg + SvnDayTestAvg + FrtnDayTestAvg, data = df.vic)

# Check OLS assumptions by inspecting the residual plots
svglite::svglite('~/COVID-19-ANALYSIS/plots/residpanel.svg')
resid_panel(linearmodel, plots = "R", smoother = TRUE, qqbands = TRUE, theme = "gray")
dev.off()

# Show and remove observations with a Cook's distance greater than 1 in the linear model 
linearmodel$model[cooks.distance(linearmodel) > 1,]
df.vic <- df.vic[-122,]



# Transforming the Model --------------------------------------------------

## Boxcox Transformation ---------------------------------------------------

# Boxcox transformation the original data frame to improve the model
df.vicx <- bind_cols(NextDayCasesx = BoxCox(df.vic$NextDayCases, lambda = BoxCox.lambda(df.vic$NextDayCases, method = 'guerrero')),
                     DailyCasesx = BoxCox(df.vic$DailyCases, lambda = BoxCox.lambda(df.vic$DailyCases, method = 'guerrero')), 
                     SvnDayCaseAvgx = BoxCox(df.vic$SvnDayCaseAvg, lambda = BoxCox.lambda(df.vic$SvnDayCaseAvg, method = 'guerrero')),
                     FrtnDayCaseAvgx = BoxCox(df.vic$FrtnDayCaseAvg, lambda = BoxCox.lambda(df.vic$FrtnDayCaseAvg, method = 'guerrero')),
                     SvnDayTestAvgx = BoxCox(df.vic$SvnDayTestAvg, lambda = BoxCox.lambda(df.vic$SvnDayTestAvg, method = 'guerrero')),
                     FrtnDayTestAvgx = BoxCox(df.vic$FrtnDayTestAvg, lambda = BoxCox.lambda(df.vic$FrtnDayTestAvg, method = 'guerrero')))

datatable(round(df.vicx, 3), options = list(pageLength =  5), caption = 'Table 3: Victoria COVID-19 Boxcox transformed data.')

# Create a new linear model with the Boxcox transformed values and then check the OLS linear regression assumptions again
linearmodelx <- lm(NextDayCasesx ~ DailyCasesx + SvnDayCaseAvgx + FrtnDayCaseAvgx + SvnDayTestAvgx + FrtnDayTestAvgx, data = df.vicx)
svglite::svglite('~/COVID-19-ANALYSIS/plots/residpanelX.svg')
resid_panel(linearmodelx, "R", smoother = TRUE, qqbands = TRUE, theme = "gray")
dev.off()

## Dimension Reduction 2 ---------------------------------------------------

# Create first linear model with all the transformed predictors and target variables
elim1 <- lm(NextDayCasesx ~ DailyCasesx + SvnDayCaseAvgx + FrtnDayCaseAvgx + SvnDayTestAvgx + FrtnDayTestAvgx, data = df.vicx) %>% 
  summary()

# Extract summary information including p values
elim1.summary <- elim1$coefficients %>% 
  round(2) %>% 
  as.data.frame() %>% 
  rownames_to_column()

elim1.summary <- elim1.summary[, c(1,5)]

# Rename and format table
col.names <- c('Predictor', 'Pr (> |t|)')
colnames(elim1.summary) <- col.names
elim1.summary %>% 
  kable(caption = 'Table 4: Backward elimination step one.')


# Create second linear model with one predictor removed
elim2 <- lm(NextDayCasesx ~ DailyCasesx + SvnDayCaseAvgx + FrtnDayCaseAvgx + SvnDayTestAvgx, data = df.vicx) %>% 
  summary()

# Extract summary information including p values
elim2.summary <- elim2$coefficients %>% 
  round(2) %>% 
  as.data.frame() %>% 
  rownames_to_column()

elim2.summary <- elim2.summary[, c(1,5)]

# Rename and format table
colnames(elim2.summary) <- col.names
elim2.summary %>% 
  kable(caption = 'Table 5: Backward elimination step two.')


# Create third linear model with one predictor removed
elim3 <- lm(NextDayCasesx ~ SvnDayCaseAvgx + FrtnDayCaseAvgx + SvnDayTestAvgx, data = df.vicx) %>% 
  summary()

# Extract summary information including p values
elim3.summary <- elim3$coefficients %>% 
  round(2) %>% 
  as.data.frame() %>% 
  rownames_to_column()

elim3.summary <- elim3.summary[, c(1,5)]

# Rename and format table
colnames(elim3.summary) <- col.names
elim3.summary %>% 
  kable(caption = 'Table 6: Backward elimination step three.')

df.vicx.optimal <- df.vicx[, c('NextDayCasesx', 'SvnDayCaseAvgx', 'FrtnDayCaseAvgx', 'SvnDayTestAvgx')]



# Supervised Machine Learning 2 -------------------------------------------

## Regression Task and Learner ---------------------------------------------

# Update the regression task with the transformed values from the df.vicx.optimal data frame
cases.taskx <- makeRegrTask(data = df.vicx.optimal, target = 'NextDayCasesx')
cases.taskx 

# Create a generalised linear regression learner
lrn <- makeLearner(cl = 'regr.glm')
lrn

## Fit Model and Evaluate Performance --------------------------------------

# Set seed value 
set.seed(1234)

# Set resample method as leave-one-out cross validation
rdesc <- makeResampleDesc(method = "LOO")

# Perform resampling of the learner on the task with leave-one-out cross validation.
model.performance <- resample(learner = lrn, task = cases.taskx, resampling = rdesc, measures = list(mae))
model.performance



# Model Limitations -------------------------------------------------------

# Split the data into random subsets
set.seed(1234)
n <- nrow(df.vicx.optimal)
train.set <- sample(n, size = 4/5*n)
test.set <- df.vicx.optimal[setdiff(1:n, train.set),]

# Train the learner on the training data set
model.optimal <- train(lrn, cases.taskx, train.set)

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
predictions <- bind_cols(prediction.points, prediction.interval)
predictions <- predictions[-2]

# Edit names
colnames(predictions) <- c('True Values', 'Predictions', '95% Lower Bound', '95% Upper Bound')

# Output table
datatable(predictions, options = list(pageLength =  5), caption = 'Table 7: Table of transformed predictions, true values, and 95% upper and lower prediction bounds.')

# Merge together the original and transformed next day case values into a new data frame
nextdaycases.compare <- bind_cols(df.vic$NextDayCases ,df.vicx.optimal$NextDayCasesx) %>% 
  round(3)

# Tidy and output merged table
colnames(nextdaycases.compare) <- c('True Values', 'Boxcox Transformed')
datatable(nextdaycases.compare, options = list(pageLength = 5), caption = 'Table 8: Mapping between Boxcox transformed values and true values of next day cases.')

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
title <- ggdraw() + draw_label("The Transformed Linear Regression Model and Mapping between Boxcox Transformed Values \nand True Values", fontface='bold')

# Output grid plot
svglite::svglite('~/COVID-19-ANALYSIS/plots/regression_mapping.svg')
plot_grid(title, plt, ncol=1, rel_heights=c(0.1, 1))
dev.off()

# Calculate and show prediction interval width
prediction.width <- ((sum(predictions$`95% Upper Bound`) - sum(predictions$`95% Lower Bound`)) / nrow(predictions)) %>% 
  round(3)
prediction.width

# Code to produce a table showing 'Boxcox Values', 'Boxcox Prediction Intervals', 'True Values', 'True Prediction Intervals' and 'True Prediction Width'.
width <- prediction.width / 2
boxcox.values <- seq(1, 10, by = 1)
true.values <- InvBoxCox(boxcox.values, lambda = BoxCox.lambda(df.vic$NextDayCases, method = 'guerrero')) %>% 
  round(0)
true.intervals <- c()
boxcox.intervals <- c()
lower.bound <- c()
upper.bound <- c()
true.width <- c()

for (i in 1:10) {
  boxcox.intervals[i] <- paste0('[', as.character(i - width), ', ', as.character(i + width), ')')
  lower.bound[i] <- InvBoxCox(i - width, lambda = BoxCox.lambda(df.vic$NextDayCases, method = 'guerrero')) %>% 
    round(0)
  upper.bound[i] <- InvBoxCox(i + width, lambda = BoxCox.lambda(df.vic$NextDayCases, method = 'guerrero')) %>% 
    round(0)
  true.intervals[i] <- paste0('[', as.character(lower.bound[i]), ', ', as.character(upper.bound[i]), ')')
  true.width[i] <- upper.bound[i] - lower.bound[i]
}

model.error <- data.frame(boxcox.values, boxcox.intervals, true.values, true.intervals, true.width)
colnames(model.error) <- c('Boxcox Values', 'Boxcox Prediction Intervals', 'True Values', 'True Prediction Intervals', 'True Prediction Width')

datatable(model.error, options = list(pageLength =  5), caption = 'Table 9: A table of prediction values and intervals to highlight the increasing width of the true prediction interval.')