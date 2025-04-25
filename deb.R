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

train_index <- createDataPartition(y, p = 0.2, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

X_train = X
y_train = y


######################## GROUP BEST SUBSET SELECTION ##########################

# (group num - 1)*18 + 1 = dove inizia il gruppo 

# s: parametro di regolarizzazione
# groups: struttura dei gruppi

# data una famiglia di modelli identificata da s, fits to the data
gbss.fit <- function(X, y, groups, s, nbest=5 ) 
{
  # tutte le possibili combinazioni di gruppi da provare
  subsets <- t(combn( 1:max(groups) , s))#togliere t
  accuracy <- c()#rimuovere
  deviance <- c()#rimuovere
  
  best_model = NULL#rimuovere
  best_dev = Inf#rimuovere
  best_subset = NULL#rimuovere
  
  
  top_models <- vector("list", nbest)
  top_deviances <- rep(Inf, nbest)
  
  for (i in 1:nrow(subsets)) 
  {
    #print(i)
    #print(subsets[i,])
    # per selezionare il sotto insieme da X
    feature_selector <- groups %in% subsets[i,]
    X_selected <- X[, feature_selector, drop = FALSE]
    model <- glm(y ~ ., data = data.frame(y, X_selected), family = binomial, singular.ok = TRUE ,control = glm.control(maxit = 1000))
    deviance[i] <-  model$deviance#rimuovere
    model$subset = subsets[i,]
    
    
    prob_predictions <- predict(model, newdata = data.frame(X_selected), type = "response")
    y_pred <- ifelse(prob_predictions > 0.5, 1, 0)#rimuovere
    accuracy[i] <-  mean(y_pred == y)#rimuovere
    
    #rimuovere
    if (model$deviance < best_dev)
    {
      best_dev = model$deviance#rimuovere
      best_model = model#rimuovere
      best_subset = subsets[i,]#rimuovere
    }
    
    
    if (model$deviance < max(top_deviances)) 
    {
      pos <- which.max(top_deviances)
      top_models[[pos]] <- model
      top_deviances[pos] <- model$deviance
    }
    
  }
  
  subsets <- cbind(subsets, accuracy)#rimuovere
  subsets <- cbind(subsets, deviance)#rimuovere
  
  # ordina dal piu piccolo al piu grande
  ord <- order(top_deviances)
  top_models <- top_models[ord]
  top_deviances <- top_deviances[ord]
  
  result <- list(
    model = top_models[[1]],
    top_models = top_models,
    top_deviances = top_deviances)
  
  return(result)
  
}

# S: l'insieme di parametri di regolarizzazione da convalida incrociata
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
      cat("s:",s,", fold:",k,"\n")
      X_train <- X[folds != k, ]
      X_test <- X[folds == k, ]
      y_train <- y[folds != k]
      y_test <- y[folds == k]
      # in teoria è data leakage fare scale all'inizio
      
      gbss_result <- gbss.fit(X_train,y_train,groups,s)
      #selected_features <- names(model$coefficients)[-1] 
      #prob_predictions <- predict(model, newdata = data.frame(X_test[, selected_features]), type = "response")
      prob_predictions <- predict(gbss_result$model, newdata = data.frame(X_test), type = "response")
      
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
#xx = cv.gbss(X,y,S=c(1,2,3,4,5), groups=feature_groups, nfolds = 10)
#model <- gbss.fit(X,y,feature_groups,xx$s.min)

######################### GBSS STABILITY SELECTION #############################
best_size = 5
groups = feature_groups

num_subsamples <- 5
subsample_size <- floor(nrow(X_train) * 0.75)
all_subsets <- list()
#TODO: replace = TRUE?
for (i in 1:num_subsamples)
{
  subsample_indices <- sample(1:nrow(X_train), subsample_size, replace = FALSE)
  X_subsample <- X_train[subsample_indices, ]
  y_subsample <- y_train[subsample_indices]
  
  #subsample_model <- glmnet(X_subsample, y_subsample, alpha = alpha, family = "binomial", lambda = lambda)
  #coefficients <- coef(subsample_model, s = lambda)[-1]
  gbss_result <- gbss.fit(X_subsample,y_subsample,groups,s=best_size)
  subset = gbss_result$model$subset
  subset_str <- paste(subset, collapse = ",")
  print(subset_str)
  all_subsets[[i]] <- subset_str
  
}

subset_counts <- sort(table(unlist(all_subsets)), decreasing = TRUE)
subset_counts
