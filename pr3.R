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
library(qgraph)
library(plotly)
library(mgcv)

rm(list=ls())
data <- read_csv("DARWIN.csv", col_names = TRUE)
data <- data[, -1]  # remove the first column (Patient ID)
y <- data[[ncol(data)]] 
X <- as.matrix(data[, -ncol(data)]) 
y <- ifelse(y == "P", 1, 0) # P -> 1, H -> 0
X = scale(X)
rm(data)
set.seed(0) 

############################ PAIRWISE CORRELATION ##############################

correlation_matrix <- cor(X)
correlation_df <- melt(correlation_matrix)
correlation_df <- correlation_df[order(-abs(correlation_df$value)), ]
correlation_df <- correlation_df[correlation_df$Var1 != correlation_df$Var2, ]
correlation_df$pair <- apply(correlation_df[, c("Var1", "Var2")], 1, function(x) paste(sort(x), collapse = "_"))
correlation_df <- correlation_df[!duplicated(correlation_df$pair), ]
correlation_df$pair <- NULL
correlation_df$abs_value = abs(correlation_df$value)
correlation_df$abs_value2 <- ifelse(correlation_df$abs_value > 0.8, correlation_df$abs_value, 0)

n_high_corr_pairs <- nrow(correlation_df[abs(correlation_df$value) > 0.8, ])
print(n_high_corr_pairs)

################################ ELASTIC NET ###################################

#............................. alpha & lambda selection ........................

#alpha: 0 ridge 1 lasso

# dato S ritornala correlazione media pesata per la sparsità
phi_ <-function(S_alpha)
{
  pi_x <- numeric(length(S_alpha))
  names(pi_x) <- S_alpha
  
  for (x in S_alpha) 
  {
    related_correlations <- correlation_df[correlation_df$Var1 == x | correlation_df$Var2 == x, ]
    related_correlations_not_selected <- related_correlations[xor(related_correlations$Var1 %in% S_alpha,related_correlations$Var2 %in% S_alpha),]
    sum_related_correlations = sum(related_correlations$abs_value)
    sum_related_correlations_not_selected = sum(related_correlations_not_selected$abs_value)
    
    pi_x[x] <- 1-ifelse(sum_related_correlations == 0, 0, sum_related_correlations_not_selected/sum_related_correlations)
    
  }
  
  pi_alpha <- ifelse(length(S_alpha) == 0, 0, sum(pi_x)/length(S_alpha))
  sigma_alpha = 1-length(S_alpha)/450
  phi_alpha = pi_alpha*sigma_alpha
  return(phi_alpha)
}

nlambda = 100
nalpha = 100
alphas = seq(from = 0, to = 0.03, length.out = nalpha)
phis <- data.frame(alpha = numeric(0), lambda = numeric(0), phi = numeric(0))

for (alpha in alphas) 
{
  elastic_net_model <- glmnet(X, y, alpha = alpha, nlambda = nlambda, family = "binomial",standardize = FALSE)
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

#.................................. Plot .......................................

grid_alpha <- seq(min(phis$alpha), max(phis$alpha), length.out = 100)
grid_lambda <- seq(min(phis$lambda), max(phis$lambda), length.out = 100)
grid_matrix <- expand.grid(alpha = grid_alpha, lambda = grid_lambda)
fit_gam <- gam(phi ~ s(alpha, lambda), data = phis)
grid_matrix$phi_pred <- predict(fit_gam, newdata = grid_matrix)
z_matrix <- matrix(grid_matrix$phi_pred, nrow = length(unique(grid_matrix$lambda)), ncol = length(unique(grid_matrix$alpha)))

plot_ly(
  data = phis,
  x = ~alpha,
  y = ~lambda,
  z = ~phi,
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 3, color = ~phi, colorscale = "Viridis", showscale = TRUE)
) %>%
  layout(
    title = "3D Scatter Plot of φ by α and λ",
    scene = list(
      xaxis = list(title = "α"),
      yaxis = list(title = "λ"),
      zaxis = list(title = "φ")
    )
  )

#............................. Stability Selection .............................

top10_rows <- phis[order(phis$phi, decreasing = TRUE), ][1:10, ]

num_features <- ncol(X)
selection_frequencies <- matrix(0, nrow = nrow(top10_rows), ncol = num_features )
num_subsamples <- 1000
subsample_size <- floor(nrow(X) / 2)

for (j in 1:nrow(top10_rows)) 
{
  row <- top10_rows[j, ]
  print(row)
  for (i in 1:num_subsamples)
  {
    subsample_indices <- sample(1:nrow(X), subsample_size, replace = FALSE)
    X_subsample <- X[subsample_indices, ]
    y_subsample <- y[subsample_indices]
    
    subsample_model <- glmnet(X_subsample, y_subsample, alpha = row$alpha, family = "binomial", lambda = row$lambda ,standardize = FALSE)
    
    coefficients <- coef(subsample_model, s = lambda)[-1]
    selected_features <- which(coefficients != 0)
    selection_frequencies[j,selected_features] <- selection_frequencies[j,selected_features] + 1
  }
}
selection_probabilities <- selection_frequencies / num_subsamples
max_selection_probabilities <- apply(selection_probabilities, 2, max)

stable_indices <- which(max_selection_probabilities >= 0.9)
feature_names <- colnames(X)
stable_feature_names <- feature_names[stable_indices]
stable_feature_names

################################ GROUP LASSO ###################################

group_stability_selection <- function(X, y, groups, num_subsamples = 100, nlam = 20, thr=0.9)
{
  data <- list(x = X, y = y)
  SGL_model <- SGL(data, groups, type="logit", standardize = FALSE, nlam = nlam ,verbose = TRUE, alpha = 0)
  lambdas <- SGL_model$lambdas
  print(lambdas)
  num_groups <- length(unique(groups))
  selection_frequencies <- matrix(0, nrow =length(lambdas) , ncol = num_groups )
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
      subsample_model <- SGL(data_subsample,groups, type="logit",lambdas = lambda, nlam = 1,standardize = FALSE, verbose = FALSE, alpha = 0)
      beta <- subsample_model$beta

      # per ogni gruppo controllo se almeno un coefficiente è non nullo
      for (gr in 1:num_groups) 
      {
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
  #print(selection_probabilities)
  #return(list(stable_groups,selection_probabilities))
  return(stable_groups)
  
}

#.............................. Task Grouping ..................................

task_groups <- rep(1:25, each = 18) 
stable_tasks = group_stability_selection(X, y, task_groups,num_subsamples = 500, nlam = 20, thr=0.9)
stable_tasks

#................................ Feature Grouping .............................

feature_groups <- rep(1:18, times = 25)
stable_features = group_stability_selection(X, y, feature_groups,num_subsamples = 500, nlam = 20, thr=0.9)
stable_features

#................................ Components Grouping ..........................

# estrai componenti
adj_matrix <- abs(correlation_matrix) > 0.8
rm(correlation_matrix)
g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", diag = FALSE)
components <- components(g)
feature_names <- colnames(X)

# stampa le componenti
for (i in 1:components$no+1)
{
  cat("Component", i-1, ":\n")
  cat(feature_names[components$membership == (i - 1)], "\n\n")
}

membership <- components$membership
names(membership) <- NULL
component_groups = membership
component_groups

stable_components = group_stability_selection(X, y, component_groups, num_subsamples = 500, nlam = 20, thr = 0.8)

for (stable_component in stable_components) 
{
  cat("Stable Component", stable_component, ":\n")
  features_in_group <- feature_names[component_groups == stable_component]
  cat(features_in_group, "\n\n")
}


######################## GROUP BEST SUBSET SELECTION ###########################

# s: parametro di regolarizzazione (numero gruppi)
# groups: struttura dei gruppi
# trova i miglior modello che include s gruppi
gbss.fit <- function(X, y, groups, s, nbest = 5 ) 
{
  # tutti i gruppi di dimensione s
  subsets <- t(combn(1:max(groups) , s))

  top_models <- vector("list", nbest)
  top_deviances <- rep(Inf, nbest)
  
  for (i in 1:nrow(subsets)) 
  {
    feature_selector <- groups %in% subsets[i,]
    X_selected <- X[, feature_selector, drop = FALSE]
    model <- glm(y ~ ., data = data.frame(y, X_selected), family = binomial, singular.ok = TRUE ,control = glm.control(maxit = 1000))
    model$subset = subsets[i,]
    
    prob_predictions <- predict(model, newdata = data.frame(X_selected), type = "response")
    
    if (model$deviance < max(top_deviances)) 
    {
      pos <- which.max(top_deviances)
      top_models[[pos]] <- model
      top_deviances[pos] <- model$deviance
    }
  }

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

# S: parametri di regolarizzazione 
cv.gbss <- function(X, y, S, groups, nfolds=3) 
{ 
  # la media dell'errori per ogni parametro
  cvm <- numeric(length(S))
  names(cvm) <- S 
  
  for (i in seq_along(S))
  {
    s <- S[i]
    folds <- sample(rep(1:nfolds, length.out = nrow(X))) 
    misclassification_rates <- numeric(nfolds)
    for (k in 1:nfolds) 
    {
      cat("s:",s,", fold:",k,"\n")
      X_train <- X[folds != k, ]
      X_test <- X[folds == k, ]
      y_train <- y[folds != k]
      y_test <- y[folds == k]

      gbss_result <- gbss.fit(X_train,y_train,groups,s)
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
task_groups <- rep(1:25, each = 18) 
xx = cv.gbss(X,y,S=c(1,2,3,4,5), groups=task_groups, nfolds = 10)

#....................... GBSS Stability Selection Variant ......................

best_size = 2
groups = task_groups

num_subsamples <- 10000
subsample_size <- floor(nrow(X) * 0.75)
all_subsets <- list()
for (i in 1:num_subsamples)
{
  print(i)
  subsample_indices <- sample(1:nrow(X), subsample_size, replace = FALSE)
  X_subsample <- X[subsample_indices, ]
  y_subsample <- y[subsample_indices]
  
  gbss_result <- gbss.fit(X_subsample,y_subsample,groups,s=best_size)
  subset = gbss_result$model$subset
  subset_str <- paste(subset, collapse = ",")
  print(subset_str)
  all_subsets[[i]] <- subset_str
  
}

subset_counts <- sort(table(unlist(all_subsets)), decreasing = TRUE)
subset_counts
