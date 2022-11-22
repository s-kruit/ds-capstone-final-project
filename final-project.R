if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(matrixStats)
library(lubridate)
library(randomForest)

# import training and validation data
training_data <- read_csv("sign_mnist_train.csv")
validation_data <- read_csv("sign_mnist_test.csv")

# get number of labels and number of rows in training and validation data sets
length(unique(training_data$label))
nrow(training_data)
nrow(validation_data)

# validation data makes up ~20% of overall data as provided. For this exercise I'd like to have a
# training data set with 90% of the overall data so we will combine both the provided train
# and test sets into one larger data set and then randomly generate new training and
# validation sets - we will still have ~3,500 rows in the validation set (>100 per label)
all_data <- bind_rows(training_data, validation_data)
set.seed(95, sample.kind = "Rounding")
test_index <- createDataPartition(y = all_data$label, times = 1, p = 0.1, list = FALSE)
training_data <- all_data[-test_index, ]
validation_data <- all_data[test_index, ]

# We will further partition the training data into train and test sets for model evaluation.
# we have an average of a bit over 1,000 rows per label - so we'll use a 50/50
# split when creating our train and test set to ensure we still have ~500 entries
# tested for each label

# separate training data into train and test sets for model testing
set.seed(9365, sample.kind = "Rounding")
test_index <- createDataPartition(y = training_data$label, times = 1, p = 0.5, list = FALSE)
train_set <- training_data[-test_index,]
test_set <- training_data[test_index,]

# ensure both train and test sets include all possible labels
all(sort(unique(train_set$label)) == sort(unique(test_set$label)))

# convert predictors to matrices for analysis
train_x <- train_set %>%
  select(-label) %>%
  as.matrix()
test_x <- test_set %>%
  select(-label) %>%
  as.matrix()
# convert labels to factors
train_y <- factor(train_set$label)
test_y <- factor(test_set$label)

# test - show image
grid <- matrix(train_x[2,], 28, 28) # the 784 predictors represent a 28x28 grid
image(1:28, 1:28, grid[, 28:1]) # image was inverted

# that sure looks like a hand signal to me!

# as the hand signals do not take up the entire image, we will analyse the variance
# for each pixel to determine whether there are pixels that have very low variance
# and can therefore be omitted without materially reducing the accuracy of our predictions
sds <- colSds(train_x)
qplot(sds, bins = "30", color = I("black"))

# this is not encouraging! Unlike previous exercises we have done there is not a large number
# of low-variance pixels we can rule out to begin with. Instead we have something resembling a
# Normal distribution. We look at an image of the variance
image(1:28, 1:28, matrix(sds, 28, 28)[, 28:1])

# there is a small area at the top centre of the image that has relatively low variance.
# But generally speaking, there are not any pixels we can easily identify as being removable
# for our analysis

# using the caret package's nearZeroVar() function confirms that it
# does not recommend any predictors for removal
nearZeroVar(train_x)

# since the previous simple variance observation wasn't useful, we will try dimension reduction
# using principal component analysis (PCA) to allow for exploratory data analysis.
time1 <- now()
pca <- prcomp(train_x)
time2 <- now()
difftime(time2, time1, units = "secs")
# took ~40 seconds
pc <- 1:784
qplot(pc, pca$sdev)

# we can see that a relatively small number of predictors account for a large proportion
# of the overall variance. Looking at the 10 highest variance principal components:
summary(pca)$importance[,1:10]

# 10 out of the 784 PCs account for nearly 70% of the variance. The table below shows
# the number of PCs required to account for different proportions of the overall
# variance:
props <- seq(0.1, 1, 0.1)
tibble(`Proportion of variance` = props,
       `PCs required` = sapply(props, function(p){
         x <- summary(pca)$importance[3,]
         sum(x < p) + 1
       }))

# as we saw in the previous table, 1 predictor accounts for more than 30% of the total
# variability, and 6 predictors account for more than 60%. This table shows we can
# even account for 80-90% of variability using only 22-59 PCs.

# we will fit models using different numbers of PCs to see what trade-offs exist between
# calculation time and accuracy

# first we transform the test set
col_means <- colMeans(test_x)
transformed_x_test <- sweep(test_x, 2, col_means) %*% pca$rotation

# we test that the dataset is balanced (so accuracy is a good metric)
train_y %>%
  table() %>%
  plot()

# we have a pretty even distribution of labels. As the data is balanced, we will
# use prediction accuracy to test the performance of our model (label 9 is not included in the dataset!)

# we try fitting a knn model using 22 and 59 PCs (corresponding with 80% and 90% of
# the total variance)

# to simplify our code, we create a function to calculate model accuracy and
# calculation time for different numbers of PCs considered
accuracy_and_time <- function(PCs, y_train = train_y, y_test = test_y,
                              x_test = transformed_x_test, x_train = pca$x){
  # function to calculate model accuracy for a given value of n PCs
  accuracy_by_PCs <- function(n){
    x_train_subset <- x_train[,1:n]
    fit <- knn3(x_train_subset, y_train)
    x_test_subset <- x_test[,1:n]
    y_hat <- predict(fit, x_test_subset, type = "class")
    confusionMatrix(y_hat, y_test)$overall["Accuracy"]
  }
  accs <- c()
  times <- c()
  # iterate through each value of n, calculate model accuracy and calculation time,
  # and add to lists
  for(n in PCs){
    time1 <- now()
    accs <- c(accs, accuracy_by_PCs(n))
    time2 <- now()
    time_elapsed <- difftime(time2, time1, units = "secs")
    times <- c(times, time_elapsed)
  }
  # return tibble of model accuracy and calc times for each k 
  tibble(PCs = PCs, `Model accuracy` = accs, `Calculation time (seconds)` = times)
}

# we run the function for 22 and 59 PCs
accuracy_and_time(c(22, 59))

# we can get a model accuracy of 0.97 in only 17 seconds using k = 59. We may be able
# to get an even higher accuracy without blowing out calculation times too much
props <- seq(0.91, 1, 0.01)
tibble(`Proportion of variance` = props,
       `PCs required` = sapply(props, function(p){
         x <- summary(pca)$importance[3,]
         sum(x < p) + 1
       }))

# we can explain 95% of the variance using ~2x the number of PCs (114) and 99% with
# 291 PCs. 

# print table and plot accuracy by number of PCs used (up to 200) - takes several minutes!!!
PC_accuracy <- accuracy_and_time(c(3, 5, 10, 25, 50, 75, 100, 125, 150, 200))
PC_accuracy
PC_accuracy %>%
  ggplot(aes(PCs, `Model accuracy`, color = `Calculation time (seconds)`)) +
  geom_point()

# once we have more than 75-100 PCs, we start to get big increases in runtime for very little improvement in model accuracy

# Next we will optimise the value of k used, using the first 22 PCs as these explain
# 80% of the predictor variance and gives us accuracy ~95% with the default k= 5, with
# calculation time of only a few seconds

# create a new function to calculate model accuracy and calculation time for different values of k
accuracy_and_time_by_k <- function(ks, n = 22, y_train = train_y, y_test = test_y,
                                   x_test = transformed_x_test, x_train = pca$x){
  # function to calculate model accuracy for given values of k
  accuracy_by_k <- function(k, n){
    x_train_subset <- x_train[,1:n]
    fit <- knn3(x_train_subset, y_train, k=k)
    x_test_subset <- x_test[,1:n]
    y_hat <- predict(fit, x_test_subset, type = "class")
    confusionMatrix(y_hat, y_test)$overall["Accuracy"]
  }
  accs <- c()
  times <- c()
  # iterate through each value of k, calculate model accuracy and calculation time,
  # and add to lists
  for(k in ks){
    time1 <- now()
    accs <- c(accs, accuracy_by_k(k, n))
    time2 <- now()
    time_elapsed <- difftime(time2, time1, units = "secs")
    times <- c(times, time_elapsed)
  }
  # return tibble of model accuracy and calc times for each k 
  tibble(k = ks, `Model accuracy` = accs, `Calculation time (seconds)` = times)
}

# we run the function to test different values of k
k_accuracy <- accuracy_and_time_by_k(c(1, 3, 5, 7, 9))
k_accuracy
k_accuracy %>%
  ggplot(aes(k, `Model accuracy`, color = `Calculation time (seconds)`)) +
  geom_point()

# Using k=1 seems to be best. This could risk over-fitting the model but running our
# prediction on the test data still gives us better accuracy with k=1 than other values
# so we will use k=1.

# now we fit a random forest model

# this approach takes way too much time - I stoped it after 12 mins
# time1 <- now()
# fit_rf <- randomForest(label~., data = training_data)
# time2 <- now()
# difftime(time2, time1, units = "secs")

# from google, using a matrix input instead of formula input (e.g. randomForest(y = example[, i], x = example[, j:k])
# instead of randomForest(y~., data = example)) can speed up the algorithm substantially - so we will do that

# we have a large data set so we will test how number of trees affects the run time and accuracy

# create a formula to test run time and accuracy for multiple values of ntree
accuracy_and_time_by_ntree <- function(ns, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y){
  accuracy_by_n <- function(n){
    set.seed(421, sample.kind = "Rounding") # we will use random seed of 421 for all random forest models we use, to ensure consistency
    fit_rf <- randomForest(x = x_train, y = y_train, ntree=n)
    y_hat <- predict(fit_rf, newdata = x_test)
    confusionMatrix(y_hat, y_test)$overall["Accuracy"]
  }
  accs <- c()
  times <- c()
  for(n in ns){
    time1 <- now()
    accs <- c(accs, accuracy_by_n(n))
    time2 <- now()
    times <- c(times, difftime(time2, time1, units = "secs"))
  }
  # return tibble of model accuracy and calc times for each ntree
  tibble(ntree = ns, `Model accuracy` = accs, `Calculation time (seconds)` = times)
}

ntree_accuracy <- accuracy_and_time_by_ntree(c(4, 8, 16, 32, 64)) # ntree=128 took 197s (accuracy 0.992)
ntree_accuracy
# we appear to get slightly better accuracy with random forest, compared to knn - albeit with a longer calculation time

# we will also try to optimise by nodesize (using ntree=8 as this gave us 0.94 accuracy in a short time)
accuracy_and_time_by_nodesize <- function(nodesizes, n=8, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y){
  accuracy_by_nodesize <- function(nodesize){
    set.seed(421, sample.kind = "Rounding") # we will use random seed of 421 for all random forest models we use, to ensure consistency
    fit_rf <- randomForest(x = x_train, y = y_train, ntree=n, nodesize = nodesize)
    y_hat <- predict(fit_rf, newdata = x_test)
    confusionMatrix(y_hat, y_test)$overall["Accuracy"]
  }
  accs <- c()
  times <- c()
  for(nodesize in nodesizes){
    time1 <- now()
    accs <- c(accs, accuracy_by_nodesize(nodesize))
    time2 <- now()
    times <- c(times, difftime(time2, time1, units = "secs"))
  }
  # return tibble of model accuracy and calc times for each ntree
  tibble(nodesize = nodesizes, `Model accuracy` = accs, `Calculation time (seconds)` = times)
}

nodesize_accuracy <- accuracy_and_time_by_nodesize(seq(1, 51, 10))
nodesize_accuracy

# accuracy diminishes as we increase nodesize above 1 - so we will leave it as is

# the random forest model so far uses all 784 predictors, leading to long calculation times.
# first we will look at variable importance
set.seed(421, sample.kind = "Rounding") # we will use random seed of 421 for all random forest models we use, to ensure consistency 
train_rf <- randomForest(x = train_x, y = train_y, ntree=16) # ntree=16 gave us 97% accuracy
imp <- tibble(pixel = 1:784, importance = importance(train_rf)) %>%
  arrange(desc(importance))

# we visualise the importance of each predictor (in terms of total decrease in node impurities as
# measured by the Gini index)
imp %>%
  mutate(n = 1, n = cumsum(n)) %>%
  ggplot(aes(n, importance)) +
  geom_point()

# from looking at the visualisation, we arbitrarily choose the top 100 most important predictors 
top100 <- imp %>%
  top_n(100, importance) %>%
  pull(pixel)

# and test the calculation times for different values of ntree using the top 100
top100_accuracy <- accuracy_and_time_by_ntree(c(4, 8, 16, 32, 64, 128, 256), x_train = train_x[, top100], x_test = test_x[, top100])
top100_accuracy

# by using the top 100 predictors we can increase ntree and get accuracy of 0.986 in less than a minute 

# This approach picks the most important predictors, but the PCA approach used earlier potentially improved
# on this by using linear transformations of multiple predictors

# we try fitting a random forest model to the first n principal components instead, and test accuracy/runtime.
# We pick ntree=8 due to relatively high accuracy and quicker calculation time
accuracy_and_time_by_PCs <- function(PCs, ntree=8, y_train = train_y, y_test = test_y,
                                     x_test=transformed_x_test, x_train=pca$x){
  # function to calculate model accuracy for a given value of n PCs
  accuracy_by_PCs <- function(n){
    x_train_subset <- x_train[, 1:n]
    set.seed(421, sample.kind = "Rounding") # we will use random seed of 421 for all random forest models we use, to ensure consistency
    fit_rf <- randomForest(x = x_train_subset, y = y_train)
    x_test_subset <- x_test[, 1:n]
    y_hat <- predict(fit_rf, x_test_subset)
    confusionMatrix(y_hat, y_test)$overall["Accuracy"]
  }
  accs <- c()
  times <- c()
  # iterate through each value of n, calculate model accuracy and calculation time,
  # and add to lists
  for(n in PCs){
    time1 <- now()
    accs <- c(accs, accuracy_by_PCs(n))
    time2 <- now()
    time_elapsed <- difftime(time2, time1, units = "secs")
    times <- c(times, time_elapsed)
  }
  # return tibble of model accuracy and calc times for each k 
  tibble(PCs = PCs, `Model accuracy` = accs, `Calculation time (seconds)` = times)
}

PC_rf_accuracy <- accuracy_and_time_by_PCs(c(5, 10, 15, 20, 25))
PC_rf_accuracy

# accuracy (and runtime) appears to be astonishingly good - ~100% accuracy in 25s using the first 20 PCs

# VALIDATION:

# now we will perform PCA on the full training data set and fit a random forest model to the top
# 20 PCs, as this was our best performing training model

# PCA on full training data set
training_x <- training_data %>%
  select(-label) %>%
  as.matrix()
time1 <- now()
final_pca <- prcomp(training_x)
time2 <- now()
difftime(time2, time1, units = "secs") # took 92s

# fit random forest model to top 20 PCs in full training data set
training_y <- factor(training_data$label)
time1 <- now()
set.seed(421, sample.kind = "Rounding")
fit_rf <- randomForest(x = final_pca$x[, 1:20], y = training_y)
time2 <- now()
difftime(time2, time1, units = "secs") # took 55s

# predict label values in the validation set using the random forest model

# prepare validation data
validation_x <- validation_data %>%
  select(-label) %>%
  as.matrix()
validation_y <- factor(validation_data$label)
validation_col_means <- colMeans(validation_x)
validation_x_transformed <- sweep(validation_x, 2, validation_col_means) %*% final_pca$rotation

# make predictions and test accuracy
y_hat <- predict(fit_rf, validation_x_transformed[, 1:20])
confusionMatrix(y_hat, validation_y)$overall["Accuracy"]

# this approach has given us ~100% accuracy with our predictions! The one potential downfall is
# the calculation time - performing PCA took a minute and a half, and then fitting the random
# forest model took another minute - an alternative approach if we are prioritising speed is to use
# the top 100 most important predictors to avoid using PCA

# We saw earlier that using the top 100 most important predictors reduced runtimes substantially -
# although actually determining the identity of those top 100 predictors took some time as we needed to first
# fit a Random Forest model (using ntree=16). To speed up this process, we will instead fit a model using
# ntree=1 to the **training_data**, hoping that the reduction in runtime will make up for any reduced accuracy
# in our determination of the top 100 most important predictors.
time1 <- now()
train_rf <- randomForest(x = training_x, y = training_y, ntree=1)
top100 <- tibble(pixel = 1:784, importance = importance(train_rf)) %>%
  arrange(desc(importance)) %>%
  top_n(100, importance) %>%
  pull(pixel)
time2 <- now()
difftime(time2, time1, units = "secs")

# This was much faster than the previous calculation. We will now test the accuracy and runtime of predictions
# made on the **validation_data** for different values of ntree.
set.seed(421, sample.kind = "Rounding")
fast_ntree_accuracy <- accuracy_and_time_by_ntree(c(4, 8, 16, 32, 64), x_train = training_x[, top100], x_test = validation_x[, top100],
                                                  y_train = training_y, y_test = validation_y)
fast_ntree_accuracy

# Using this approach we could achieve prediction accuracy of 0.94 with less than 10 seconds of total runtime
# (identifying the most important predictors and then fitting the Random Forest model to those predictors), or
# accuracy of 0.99 in under 15 seconds. While our Model 6 prediction achieved the highest accuracy, it is helpful
# to have other faster options in case a particular use case requires faster turnaround and is willing to accept
# a slightly lower level of accuracy.
