library(glmnet)
library(caret)
library(readr)
library(gglasso)
library(SGL)
library(sparsepca)
library(reshape2)
library(leaps)
library(igraph)
library(glasso)
library(ggplot2)
library(readr)
library(glasso)
library(igraph)


rm(list=ls())
set.seed(0) 


data <- read_csv("DARWIN.csv", col_names = TRUE)
data <- data[, -1]  # Remove the first column (Patient ID)

y <- as.factor(data[[ncol(data)]])  # Last column as response variable
X <- as.matrix(data[, -ncol(data)]) # All other columns as predictors
y <- as.factor(ifelse(y == "P", 1, 0))  # "P" → 1, "H" → 0

X <- scale(X)

train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]


# (group num - 1)*18 + 1 = dove inizia il gruppo 

# s: parametro di regolarizzazione
# groups: struttura dei gruppi

# data una famiglia di modelli identificata da s, fits to the data
gbss.fit <- function(X, y, groups, s ) 
{
  # tutte le possibili combinazioni di gruppi da provare
  subsets <- t(combn( 1:max(groups) , s))#togliere t
  accuracy <- c()#rimuovere
  deviance <- c()#rimuovere
  
  best_model = NULL
  best_dev = Inf#rimuovere
  best_subset = NULL#rimuovere
  
  for (i in 1:nrow(subsets)) 
  {
    print(i)
    print(subsets[i,])
    # per selezionare il sotto insieme da X
    feature_selector <- groups %in% subsets[i,]
    X_selected <- X[, feature_selector, drop = FALSE]
    model <- glm(y ~ ., data = data.frame(y, X_selected), family = binomial, singular.ok = TRUE ,control = glm.control(maxit = 1000))

    deviance[i] <-  model$deviance#rimuovere
    
    prob_predictions <- predict(model, newdata = data.frame(X_selected), type = "response")
    y_pred <- ifelse(prob_predictions > 0.5, 1, 0)#rimuovere
    accuracy[i] <-  mean(y_pred == y)#rimuovere
    
    if (model$deviance < best_dev)
    {
      best_dev = model$deviance#rimuovere
      best_model = model
      best_subset = subsets[i,]#rimuovere
    }
    
  }
  
  subsets <- cbind(subsets, accuracy)#rimuovere
  subsets <- cbind(subsets, deviance)#rimuovere
  

  return (best_model)
  
  
}
# S: l'insieme di parametri di regolarizzazione da convalidazione incrociata
cv.gbss <- function(X,y,S,groups,nfolds=3) 
{ 
  # la media dell'errori per ogni parametro
  cvm <- numeric(length(S))
  names(cvm) <- S 
  
  for (i in seq_along(S))
  {
    s <- S[i]
    folds <- sample(rep(1:nfolds, length.out = nrow(X))) 
    # errori medi per ogni fold fissato s
    misclassification_rates <- numeric(nfolds)
    for (k in 1:nfolds) 
    {
      X_train <- X[folds != k, ]
      X_test <- X[folds == k, ]
      y_train <- y[folds != k]
      y_test <- y[folds == k]
      
      model <- gbss.fit(X_train,y_train,groups,s)
      #selected_features <- names(model$coefficients)[-1] 
      #prob_predictions <- predict(model, newdata = data.frame(X_test[, selected_features]), type = "response")
      prob_predictions <- predict(model, newdata = data.frame(X_test), type = "response")
      
      y_pred <- ifelse(prob_predictions > 0.5, 1, 0)
      misclassification_rates[k] <- mean(y_pred != y_test)
    }
    
    cvm[i] <- mean(misclassification_rates)
  }
  
  s.min <- S[which.min(cvm)]
  return(list(cvm = cvm, s.min = s.min))
  
}
feature_groups <- rep(1:18, times = 25)
task_goups <- rep(1:25, each = 18) 
xx = cv.gbss(X,y,c(1,2),feature_groups)

model <- gbss.fit(X,y_train,feature_groups,xx$s.min)


