{%- if cookiecutter.framework == 'keras' %}
# Based on:
# https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_classification/

# --------------------------------------------------
# To get started with Comet and R, please see:
# https://www.comet.ml/docs/r-sdk/getting-started/
#
# Specifically, you need to create a .comet.yml file
# or add your Comet API key to create_experiment()
# --------------------------------------------------

# install.packages("cometr")
# devtools::install_github("rstudio/keras")
# install.packages("tidyr")

# library(tensorflow)
# install_keras()
{%- endif %}
{%- if cookiecutter.framework == 'caret' %}
# https://github.com/RickPack/R-Dojo/blob/master/RDojo_MachLearn.R

# Created by Rick Pack and Chad Himes during the Triangle .NET User Group
#   "Introduction to R" dojo, led by
#   Kevin Feasel and Jamie Dixon.
# Vast majority of code from the tutorial by Jason Brownlee:
# "Your First Machine Learning Project in R
#   Step-By-Step (tutorial and template for future projects)"
#    http://machinelearningmastery.com/machine-learning-in-r-step-by-step/

# --------------------------------------------------
# To get started with Comet and R, please see:
# https://www.comet.ml/docs/r-sdk/getting-started/
#
# Specifically, you need to create a .comet.yml file
# or add your Comet API key to create_experiment()
# --------------------------------------------------

#install.packages("cometr")
#install.packages("caret")
#install.packages("ellipse")
{%- endif %}
{%- if cookiecutter.framework == 'nnet' %}
# Based on:
# https://www.rdocumentation.org/packages/nnet/versions/7.3-13/topics/nnet

# --------------------------------------------------
# To get started with Comet and R, please see:
# https://www.comet.ml/docs/r-sdk/getting-started/
#
# Specifically, you need to create a .comet.yml file
# or add your Comet API key to create_experiment()
# --------------------------------------------------

#install.packages("cometr")
{%- endif %}

# Libraries:
library(cometr)

{%- if cookiecutter.framework == 'keras' %}
library(tidyr)
library(ggplot2)
library(keras)
library(reticulate)

{%- endif %}
{%- if cookiecutter.framework == 'caret' %}
library(caret)
library(ellipse)

{%- endif %}
{%- if cookiecutter.framework == 'nnet' %}
library(nnet)
library(stringr)

{%- endif %}

exp <- create_experiment(
  keep_active = TRUE,
  log_output = TRUE,
  log_error = FALSE,
  log_code = TRUE,
  log_system_details = TRUE,
  log_git_info = TRUE
)

exp$add_tags(c("made with {{cookiecutter.framework}}"))


{%- if cookiecutter.framework == 'nnet' %}
# sample the iris data:

sample_size <- 25 # of each iris type
total_size <- 50
total_size2 <- total_size * 2

exp$log_parameter("sample_size", sample_size)

ir <- rbind(iris3[,,1], iris3[,,2], iris3[,,3])
targets <- class.ind(c(
  rep("s", total_size),
  rep("c", total_size),
  rep("v", total_size))
)
samp <- c(
  sample(1:total_size, sample_size),
  sample((total_size + 1):(total_size * 2), sample_size),
  sample(((total_size * 2) + 1):(total_size * 3), sample_size)
)

weight_decay <- 5e-4
epochs <- 200
hidden_layer_size <- 2
initial_random_weight_range <- 0.1

exp$log_parameter("weight_decay", weight_decay)
exp$log_parameter("epochs", epochs)
exp$log_parameter("hidden_layer_size", hidden_layer_size)
exp$log_parameter("initial_random_weight_range", initial_random_weight_range)

ir1 <- NULL

train <- function() {
  ir1 <<- nnet(
    ir[samp,],
    targets[samp,],
    size = hidden_layer_size,
    rang = initial_random_weight_range,
    decay = weight_decay,
    maxit = epochs)
    ir1
}

output <- capture.output(train(), split = TRUE)
output <- strsplit(output, "\n")

# "initial  value 57.703088 "
for (match in str_match(output, "^initial\\s+value\\s+([-+]?[0-9]*\\.?[0-9]+)")[,2]) {
  if (!is.na(match)) {
     exp$log_metric("loss", match, step=0)
  }
}

# "iter  10 value 46.803951"
matrix = str_match(output, "^iter\\s+(\\d+)\\s+value\\s+([-+]?[0-9]*\\.?[0-9]+)")
for (i in 1:nrow(matrix)) {
  match = matrix[i,]
  if (!is.na(match[2])) {
     exp$log_metric("loss", match[3], step=match[2])
  }
}

test.cl <- function(true, pred) {
    true <- max.col(true)
    cres <- max.col(pred)
    table(true, cres)
}
cm = test.cl(targets[-samp,], predict(ir1, ir[-samp,]))

matrix <- sprintf("[%s,%s,%s]",
                  sprintf("[%s]", paste(cm[1,], collapse=",")),
                  sprintf("[%s]", paste(cm[2,], collapse=",")),
                  sprintf("[%s]", paste(cm[3,], collapse=",")))

title <- "Iris Confusion Matrix"
labels <- sprintf('["%s","%s","%s"]', "Setosa", "Versicolor", "Virginica")

template <- '{"version":1,"title":"%s","labels":%s,"matrix":%s,"rowLabel":"Actual Category","columnLabel":"Predicted Category","maxSamplesPerCell":25,"sampleMatrix":[],"type":"integer"}'

fp <- file("confusion_matrix.json")
writeLines(c(sprintf(template, title, labels, matrix)), fp)
close(fp)

exp$upload_asset("confusion_matrix.json", type = "confusion-matrix")

# Now log some HTML:

exp$log_html("
<h1>Comet nnet Example</h1>

<p>This example demonstrates using the nnet library on the iris dataset.</p>

<p>See the Output tab for confusion matrix.</p>

<ul>
<li><a href=https://github.com/comet-ml/cometr/blob/master/inst/train-examples/nnet-example.R>github.com/comet-ml/cometr/inst/train-example/nnet-example.R</a></li>
</ul>

<p>For help on the Comet R SDK, please see: <a href=https://www.comet.ml/docs/r-sdk/getting-started/>www.comet.ml/docs/r-sdk/getting-started/</a></p>
")

{%- endif %}
{%- if cookiecutter.framework == 'keras' %}
fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

train_images <- train_images / 255
test_images <- test_images / 255

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

epochs <- 5

exp$log_parameter("epochs", epochs)

LogMetrics <- R6::R6Class("LogMetrics",
  inherit = KerasCallback,
  public = list(
    losses = NULL,
    on_epoch_end = function(epoch, logs = list()) {
      # Had trouble logging directly to exp here
      # so we do it when we get back in R
      self$losses <- c(self$losses, c(epoch, logs[["loss"]]))
    }
  )
)

callback <- LogMetrics$new()

model %>% fit(train_images, train_labels, epochs = epochs, verbose = 2,
      callbacks = list(callback))

# Log the losses now:
losses <- matrix(callback$losses, nrow = 2)
for (i in 1:ncol(losses)) {
  exp$log_metric("loss", losses[2, i], step=losses[1, i])
}

score <- model %>% evaluate(test_images, test_labels, verbose = 0)

cat('Test loss:', score$loss, "\n")

cat('Test accuracy:', score$acc, "\n")

exp$log_metric("test_loss", score$loss)
exp$log_metric("test_accuracy", score$acc)

predictions <- model %>% predict(test_images)

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

png(file = "FashionMNISTResults.png")

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev))
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800'
  } else {
    color <- '#bb0000'
  }

  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}

dev.off()
exp$upload_asset("FashionMNISTResults.png")

{%- endif %}
{%- if cookiecutter.framework == 'caret' %}
# attach the iris dataset to the environment
data(iris)

# rename the dataset
dataset <- iris

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

# split input and output
# 5th column is outcome variable "Species"
# 1-4th are two types of length and width measurements
x <- dataset[,1:4]
y <- dataset[,5]
# scatterplot matrix
# Scatterplot shows overlap of green (species "virginica")
#   and pink (species "versicolor")

png(file = "FeaturePlot.png")
featurePlot(x=x, y=y, plot="ellipse")
dev.off()
exp$upload_asset("FeaturePlot.png")

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(7)

exp$log_parameter("seed", 7)

fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

## b) nonlinear algorithms
## CART
#set.seed(7)
#fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
## kNN
#set.seed(7)
#fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
## c) advanced algorithms
## SVM
#set.seed(7)
#fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
## Random Forest
#set.seed(7)
#fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

# summarize accuracy of models
#results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
#summary(results)

# estimate skill of LDA on the validation dataset
# Shows accuracy is 100% (1) for validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

exp$log_html("
<h1>Comet Caret Example</h1>

<p>This example demonstrates using the caret library on the iris dataset.</p>

<p>See the Output tab for confusion matrix.</p>

<ul>
<li><a href=https://github.com/comet-ml/cometr/blob/master/inst/train-examples/caret-example.R>github.com/comet-ml/cometr/inst/train-example/caret-example.R</a></li>
</ul>

<p>For help on the Comet R SDK, please see: <a href=https://www.comet.ml/docs/r-sdk/getting-started/>www.comet.ml/docs/r-sdk/getting-started/</a></p>

")
{%- endif %}


exp$log_other(key = "Created by", value = "cometr")
exp$print()
exp$stop()
