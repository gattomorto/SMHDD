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



#perchè ce due volta as factor?
rm(list=ls())
data <- read_csv("DARWIN.csv", col_names = TRUE)
data <- data[, -1]  # remove the first column (Patient ID)
y <- data[[ncol(data)]] 
X <- as.matrix(data[, -ncol(data)]) 
#y <- as.factor(ifelse(y == "P", 1, 0))  # "P" → 1, "H" → 0
y <- ifelse(y == "P", 1, 0) # "P" → 1, "H" → 0

set.seed(0) 

############################ PAIRWISE CORRELATION ##############################

correlation_matrix <- cor(X)
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


n_high_corr_pairs <- nrow(correlation_df[abs(correlation_df$value) > 0.8, ])
print(n_high_corr_pairs)

#X_train <- as.data.frame(X_train)
#plot(X_train$max_x_extension1 , X_train$max_y_extension1 )
#cor(X_train$max_x_extension1, X_train$max_y_extension1)



################################ CLUSTER ANALYSIS ######################################

cor_matrix <- cor(X)
cor_matrix <- cor_matrix[1:100, 1:100]

abs_cor_matrix = abs(cor_matrix)

threshold <- 0.7
cor_matrix[abs_cor_matrix < threshold] <- 0  # Set values below threshold to NA

#-----heat map-------oooooooooooooooooooooooooooooooooooooooo

# Heatmap of the correlation matrix
library(ggplot2)
library(reshape2)

# Melt the correlation matrix for plotting
melted_cor <- melt(cor_matrix)

# Create a heatmap
ggplot(data = melted_cor, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ggtitle("Correlation Heatmap")

#hiecal clustering--oooooooooooooooooooooooooooooooooooooooo

# Perform hierarchical clustering on the correlation matrix
distance <- as.dist(1 - cor_matrix)  # Convert correlation to distance
hc <- hclust(distance, method = "ward.D2")

# Plot the dendrogram
plot(hc, main = "Variable Clustering", xlab = "", sub = "")

# Cut the dendrogram into k groups
k <- 3  # Number of groups
groups <- cutree(hc, k = k)

# Add group labels to the original variables
grouped_vars <- data.frame(Variable = rownames(cor_matrix), Group = groups)
print(grouped_vars)

#hierchal clustering 2---oooooooooooooooooooooooooooooooooooooooo

library(Hmisc)
# Compute variable clusters
clusters <- varclus(X_train, similarity = "spearman")  # or "pearson"
plot(clusters)  # Dendrogram showing clusters
summary(clusters)  # Details of each cluster

#hierchal clust 3---oooooooooooooooooooooooooooooooooooooooo

plot(hclust(as.dist(1-abs(cor(na.omit(X_train))))))

#block heatmap 1 ---oooooooooooooooooooooooooooooooooooooooo

dist_matrix <- as.dist(1 - abs(cor_matrix))  # Convert correlation to distance (1 - |correlation|)
clustering <- hclust(dist_matrix, method = "complete")  # Hierarchical clustering

library(ggcorrplot)

# Reorder the correlation matrix based on clustering
ordered_corr_matrix <- cor_matrix[clustering$order, clustering$order]

ggcorrplot(ordered_corr_matrix, hc.order = TRUE, type = "lower", lab = FALSE)

#block heatmap 2---oooooooooooooooooooooooooooooooooooooooo


#alternativamente
library(pheatmap)
pheatmap((cor_matrix), clustering_method = "complete")  







################## ALPHA LAMBDA SELECTION ELASTIC NET ##########################
#0 ridge 1 lasso

#dato S ritornala correlazione media pesata per la sparsità
phi_ <-function(S_alpha)
{
  pi_x <- numeric(length(S_alpha))
  names(pi_x) <- S_alpha
  
  # For each selected variable, calculate the sum of absolute correlations
  for (x in S_alpha) 
  {
    related_correlations <- correlation_df[correlation_df$Var1 == x | correlation_df$Var2 == x, ]
    related_correlations_not_selected <- related_correlations[
      xor(related_correlations$Var1 %in% S_alpha, related_correlations$Var2 %in% S_alpha), 
    ]
    sum_related_correlations = sum(related_correlations$abs_value)
    sum_related_correlations_not_selected = sum(related_correlations_not_selected$abs_value)
    
    pi_x[x] <- 1-ifelse(sum_related_correlations == 0, 0, sum_related_correlations_not_selected/sum_related_correlations)
    
  }
  
  pi_alpha <- ifelse(length(S_alpha) == 0, 0, sum(pi_x)/length(S_alpha))
  sigma_alpha = 1-length(S_alpha)/450
  phi_alpha = pi_alpha*sigma_alpha
  return(phi_alpha)
}

nlambda = 50
nalpha = 50
alphas = seq(from = 0, to = 0.003, length.out = nalpha)
#alphas=10^seq(log10(1e-6), log10(1), length.out = nalpha)
alphas = alphas[-1]
phis <- data.frame(alpha = numeric(0), lambda = numeric(0), phi = numeric(0))


for (alpha in alphas) 
{
  elastic_net_model <- glmnet(X, y, alpha = alpha, nlambda = nlambda, family = "binomial",standardize = TRUE)
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

top10_rows <- phis[order(phis$phi, decreasing = TRUE), ][1:10, ]


#           alpha   lambda       phi
#    0.0006122449 191.7164 0.3340376


########################
grid_alpha <- seq(min(phis$alpha), max(phis$alpha), length.out = 100)
grid_lambda <- seq(min(phis$lambda), max(phis$lambda), length.out = 100)
grid_matrix <- expand.grid(alpha = grid_alpha, lambda = grid_lambda)


fit_gam <- gam(phi ~ s(alpha, lambda), data = phis)

grid_matrix$phi_pred <- predict(fit_gam, newdata = grid_matrix)


#######################################################


# Reshape the predicted surface for plotting
z_matrix <- matrix(grid_matrix$phi_pred, 
                   nrow = length(unique(grid_matrix$lambda)), 
                   ncol = length(unique(grid_matrix$alpha)))

# 3D plot with surface and original data points
plot_ly() %>%
  add_surface(
    x = ~unique(grid_matrix$alpha),
    y = ~unique(grid_matrix$lambda),
    z = ~z_matrix,
    colorscale = "Viridis",
    showscale = TRUE
  ) %>%
  add_markers(
    data = phis,
    x = ~alpha,
    y = ~lambda,
    z = ~phi,
    marker = list(size = 2, color = "black"),
    name = "Original φ"
  ) %>%
  layout(
    title = "Smoothed φ Surface with Original Points",
    scene = list(
      xaxis = list(title = "α"),
      yaxis = list(title = "λ"),
      zaxis = list(title = "φ")
    )
  )

########################


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


##################### ELASTIC NET STABILITY SELECTION ##########################

top10_rows <- phis[order(phis$phi, decreasing = TRUE), ][1:10, ]

num_features <- ncol(X)
selection_frequencies <- matrix(0, nrow = nrow(top10_rows) , ncol =num_features )
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
    
    subsample_model <- glmnet(X_subsample, y_subsample, alpha = row$alpha, family = "binomial", lambda = row$lambda ,standardize = TRUE)
    
    coefficients <- coef(subsample_model, s = lambda)[-1]
    # selected_features <- which(abs(coefficients) > 1e-6)?
    selected_features <- which(coefficients != 0)
    #print(selected_features)
    selection_frequencies[j,selected_features] <- selection_frequencies[j,selected_features] + 1
  }
}
selection_probabilities <- selection_frequencies / num_subsamples
max_selection_probabilities <- apply(selection_probabilities, 2, max)

stable_indices <- which(max_selection_probabilities >= 0.9)
feature_names <- colnames(X)
stable_feature_names <- feature_names[stable_indices]
stable_feature_names


################### GROUP LASSO PER TASK STABILITY SELECTION ###################
#intercept dov'è?
group_stability_selection <- function(X,y,groups, num_subsamples = 100, nlam = 20 ,thr=0.9)
{
  y <- ifelse(y == 1, 1, 0)
  
  data <- list(x = X, y = y)
  SGL_model <- SGL(data, groups, type="logit",standardize = TRUE, nlam = nlam ,verbose = TRUE, alpha = 0)
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



################## GROUP LASSO PER FEATURE STABILITY SELECTION #################



groups <- rep(1:18, times = 25)




################ GROUP LASSO ON COMPONENTS STABILITY SELECTION #################

# estrai componenti
corr_matrix <- cor(X)
adj_matrix <- abs(corr_matrix) > 0.7
g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", diag = FALSE)
components <- components(g)
feature_names <- colnames(X)

# stampa le componenti
#sink("components_output1.txt")
for (i in 1:components$no+1)
{
  cat("Component", i-1, ":\n")
  cat(feature_names[components$membership == (i - 1)], "\n\n")
}
#sink()

membership <- components$membership
names(membership) <- NULL
membership
groups = membership
groups

for (stable_group in stable_groups) {
  cat("Stable Group", stable_group, ":\n")
  features_in_group <- feature_names[groups == stable_group]
  cat(features_in_group, "\n\n")
}


######################## GROUP BEST SUBSET SELECTION ##########################
#TODO: 𝜷̂𝐵𝑆𝑆 = arg min ||𝐲 − 𝐗𝜷||2 s.t. ||𝜷||0 ≤ 𝑘
#TODO: siamo sicuri che non ci vuole qualche shuffling?

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
#X <- scale(X)# credo che sia solo per la velocità
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
  
  gbss_result <- gbss.fit(X_subsample,y_subsample,groups,s=best_size)
  subset = gbss_result$model$subset
  subset_str <- paste(subset, collapse = ",")
  print(subset_str)
  all_subsets[[i]] <- subset_str
  
}

subset_counts <- sort(table(unlist(all_subsets)), decreasing = TRUE)
subset_counts