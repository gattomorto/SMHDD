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
data <- read_csv("DARWIN.csv", col_names = TRUE)
data <- data[, -1]  # Remove the first column (Patient ID)
y <- as.factor(data[[ncol(data)]]) 
X <- as.matrix(data[, -ncol(data)]) 
y <- as.factor(ifelse(y == "P", 1, 0))  # "P" → 1, "H" → 0
X <- scale(X)
set.seed(0) 


################################################################################
#intercept dov'è?
group_stability_selection <- function(X,y,groups, num_subsamples = 100, nlam = 20 ,thr=0.9)
{
  y <- ifelse(y == 1, 1, 0)
  
  data <- list(x = X, y = y)
  SGL_model <- SGL(data,groups, type="logit",standardize = TRUE, nlam = nlam ,verbose = TRUE, alpha = 0)
  lambdas <- SGL_model$lambdas
  print(lambdas)
  num_groups <- length(unique(groups))
  selection_frequencies <- matrix(0, nrow =length(lambdas) , ncol = num_groups )
  #num_subsamples <- 100
  subsample_size <- floor(nrow(X) / 2)
  
  for (lambda_idx in seq_along(lambdas)) 
  {
    lambda <- lambdas[lambda_idx]
    print(lambda_idx)
    for (i in 1:num_subsamples)
    {
      subsample_indices <- sample(1:nrow(X), subsample_size, replace = FALSE)
      X_subsample <- X[subsample_indices, ]
      y_subsample <- y[subsample_indices]
      data_subsample <- list(x = X_subsample, y = y_subsample)
      subsample_model <- SGL(data_subsample,groups, type="logit",lambdas = lambda,nlam = 1,standardize = TRUE, verbose = FALSE, alpha = 0)
      beta <- subsample_model$beta
      
      # per ogni gruppo controllo se almeno un coefficiente è non nullo
      for (gr in 1:num_groups) 
      {
        # tutti i coefficienti del gruppo gr (lunghezza 18)
        beta_gr <- beta[groups == gr]
        if (any(beta_gr != 0)) 
        {
          selection_frequencies[lambda_idx,gr] <- selection_frequencies[lambda_idx,gr] + 1
        }
      }
    }
  }
  selection_probabilities <- selection_frequencies / num_subsamples
  max_selection_probabilities <- apply(selection_probabilities, 2, max)
  max_selection_probabilities
  stable_groups <- which(max_selection_probabilities >= thr)
  return(stable_groups)
}

groups <- rep(1:25, each = 18) 
groups <- rep(1:18, times = 25)

sg = group_stability_selection(X,y,groups, num_subsamples = 20, nlam = 2)
sg
