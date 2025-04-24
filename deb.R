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


##################
correlation_matrix <- cor(X_train)
# perchè la matrice ha la metà delle informazioni rindondanti
correlation_df <- melt(correlation_matrix)
correlation_df <- correlation_df[order(-abs(correlation_df$value)), ]
correlation_df <- correlation_df[correlation_df$Var1 != correlation_df$Var2, ]
correlation_df$pair <- apply(correlation_df[, c("Var1", "Var2")], 1, function(x) paste(sort(x), collapse = "_"))
correlation_df <- correlation_df[!duplicated(correlation_df$pair), ]
correlation_df$pair <- NULL
correlation_df$abs_value = abs(correlation_df$value)
correlation_df$abs_value2 <- ifelse(correlation_df$abs_value > 0.8,
                                    correlation_df$abs_value,
                                    0)
###################
#0 ridge 1 lasso

#dato S ritornala correlazione media pesata per la sparsità
phi_ <-function(S_alpha)
{
  pi_x <- numeric(length(S_alpha))
  names(pi_x) <- S_alpha
  
  # For each selected variable, calculate the sum of absolute correlations
  for (x in S_alpha) 
  {
    #x = "X1"
    related_correlations <- correlation_df[correlation_df$Var1 == x | correlation_df$Var2 == x, ]
    related_correlations_not_selected <- related_correlations[
      xor(related_correlations$Var1 %in% S_alpha, related_correlations$Var2 %in% S_alpha), 
    ]
    sum_related_correlations = sum(related_correlations$abs_value2)
    sum_related_correlations_not_selected = sum(related_correlations_not_selected$abs_value2)
    
    pi_x[x] <- 1-ifelse(sum_related_correlations == 0, 0, sum_related_correlations_not_selected/sum_related_correlations)
    
  }
  
  pi_alpha <-ifelse(length(S_alpha) == 0, 0, sum(pi_x)/length(S_alpha))
  sigma_alpha = 1-length(S_alpha)/450
  phi_alpha = pi_alpha*sigma_alpha
  return(phi_alpha)
}

nlambda = 5
nalpha = 5
alphas = seq(from = 0, to = 1, length.out = nalpha)
alphas = alphas[-1]
phis <- data.frame(alpha = numeric(0), lambda = numeric(0), phi = numeric(0))

for (alpha in alphas) 
{
  elastic_net_model <- glmnet(X_train, y_train, alpha = alpha, nlambda = nlambda, family = "binomial")
  lambdas <- elastic_net_model$lambda
  for (lambda in lambdas)
  {
    coef_enet <- coef(elastic_net_model,s = lambda)
    coef_values <- as.vector(coef_enet[-1])  
    names(coef_values) <- rownames(coef_enet)[-1] 
    S_alpha <- names(coef_values[coef_values != 0])
    phi = phi_(S_alpha = S_alpha)
    phis <- rbind(phis, data.frame(alpha = alpha, lambda = lambda, phi = phi))
    cat("alpha:",alpha,"lambda:",lambda,  "phi:", phi,"\n")
  }

}
max_row <- phis[which.max(phis$phi), ]
max_row
#         alpha    lambda       phi
# 103 0.1666667 0.9457503 0.7128748


