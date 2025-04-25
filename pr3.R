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

rm(list=ls())

data <- read_csv("DARWIN.csv", col_names = TRUE)
data <- data[, -1]  # Remove the first column (Patient ID)
y <- as.factor(data[[ncol(data)]]) 
X <- as.matrix(data[, -ncol(data)]) 
y <- as.factor(ifelse(y == "P", 1, 0))  # "P" → 1, "H" → 0
X <- scale(X)
set.seed(0) 

############################ PAIRWISE CORRELATION ##############################

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


n_high_corr_pairs <- nrow(correlation_df[abs(correlation_df$value) > 0.8, ])
print(n_high_corr_pairs)

#X_train <- as.data.frame(X_train)
#plot(X_train$max_x_extension1 , X_train$max_y_extension1 )
#cor(X_train$max_x_extension1, X_train$max_y_extension1)


# Nota che qst non mostra tutte, solo le variabili da eliminare
#high_corr_vars <- findCorrelation(correlation_matrix, cutoff = 0.9997608, names = TRUE)
#print(high_corr_vars)



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







############################### GRAPHICAL LASSO ################################

rm(list=ls())

data <- read_csv("DARWIN.csv", col_names = TRUE)
data <- data[, -1]  # Remove the first column (Patient ID)
y <- as.factor(data[[ncol(data)]])  
X <- as.matrix(data[, -ncol(data)])
y <- as.factor(ifelse(y == "P", 1, 0))  
X_1 <- X[y == 1, ]  # Subset where y = 1
X_0 <- X[y == 0, ]  # Subset where y = 0

X_1_scaled <- scale(X_1)
X_0_scaled <- scale(X_0)
S_1 <- cov(X_1_scaled)
S_0 <- cov(X_0_scaled)
lambda = 0.1
glasso_1 <- glasso(S_1, rho = lambda, trace = TRUE)
glasso_0 <- glasso(S_0, rho = lambda, trace = TRUE)
precision_1 <- glasso_1$wi
precision_0 <- glasso_0$wi
adj_1 <- ifelse(abs(precision_1) > 1e-6, 1, 0)
diag(adj_1) <- 0
adj_0 <- ifelse(abs(precision_0) > 1e-6, 1, 0)
diag(adj_0) <- 0
graph_1 <- graph_from_adjacency_matrix(adj_1, mode = "undirected")
graph_0 <- graph_from_adjacency_matrix(adj_0, mode = "undirected")

#degrees
degrees_1 <- degree(graph_1)
names(degrees_1) <- colnames(X)
degrees_1_sorted <- sort(degrees_1)
degrees_1_sorted
degrees_0 <- degree(graph_0)
names(degrees_0) <- colnames(X)
degrees_0_sorted <- sort(degrees_0)
degrees_0_sorted


num_isolated_0 <- sum(degrees_0 == 0)
print(paste("Number of isolated nodes:", num_isolated_0))
num_isolated_1 <- sum(degrees_1 == 0)
print(paste("Number of isolated nodes:", num_isolated_1))



#graph difference
diff_adj <- adj_1 - adj_0
diff_edges <- which(diff_adj != 0, arr.ind = TRUE)
node_names <- colnames(X)
diff_edges_df <- data.frame(Node1 = node_names[diff_edges[, 1]], 
                            Node2 = node_names[diff_edges[, 2]], 
                            Change = diff_adj[diff_edges])
print(diff_edges_df)





community_1 <- cluster_louvain(graph_1)
community_0 <- cluster_louvain(graph_0)
print(membership(community_1))
print(membership(community_0))
par(mfrow = c(1, 2))  # Side-by-side plotting
plot(graph_1, vertex.color = membership(community_1), 
     main = "Community Detection (y=1)", vertex.label = NA)
plot(graph_0, vertex.color = membership(community_0), 
     main = "Community Detection (y=0)", vertex.label = NA)
num_communities_1 <- length(unique(membership(community_1)))
num_communities_0 <- length(unique(membership(community_0)))


community_sizes_1 <- sizes(community_1)
community_sizes_1
community_sizes_0 <- sizes(community_0)
community_sizes_0
largest_community_1 <- which.max(community_sizes_1)
largest_community_1
largest_community_0 <- which.max(community_sizes_0)
largest_community_0
cat("Largest community for y=1:", largest_community_1, "with size:", community_sizes_1[largest_community_1], "\n")
cat("Largest community for y=0:", largest_community_0, "with size:", community_sizes_0[largest_community_0], "\n")



# connected_graph_1 <- delete.vertices(graph_1, which(degrees_1 == 0))
# plot(connected_graph_1,
#      #vertex.label = colnames(X_scaled)[node_degrees != 0],  # Keep only non-isolated labels
#      vertex.size = 10,
#      vertex.color = "lightblue",
#      edge.color = "gray",
#      main = "Graphical Lasso Network (No Isolated Nodes)")









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
##################### ELASTIC NET STABILITY SELECTION ##########################
alpha = 0.16
lambda = 
elastic_net_model <- glmnet(X_train, y_train, alpha = alpha, family = "binomial")
lambdas <- elastic_net_model$lambda

num_features <- ncol(X_train)
selection_frequencies <- matrix(0, nrow =length(lambdas) , ncol =num_features )
num_subsamples <- 100
subsample_size <- floor(nrow(X_train) / 2)

for (lambda_idx in seq_along(lambdas)) 
{
  lambda <- lambdas[lambda_idx]
  print(lambda_idx)
  for (i in 1:num_subsamples)
  {
    subsample_indices <- sample(1:nrow(X_train), subsample_size, replace = FALSE)
    X_subsample <- X_train[subsample_indices, ]
    y_subsample <- y_train[subsample_indices]
    
    subsample_model <- glmnet(X_subsample, y_subsample, alpha = alpha, family = "binomial", lambda = lambda)
    
    coefficients <- coef(subsample_model, s = lambda)[-1]
    # selected_features <- which(abs(coefficients) > 1e-6)?
    selected_features <- which(coefficients != 0)
    selection_frequencies[lambda_idx,selected_features] <- selection_frequencies[lambda_idx,selected_features] + 1
  }
}
selection_probabilities <- selection_frequencies / num_subsamples
max_selection_probabilities <- apply(selection_probabilities, 2, max)

stable_indices <- which(max_selection_probabilities >= 0.8)
feature_names <- colnames(X_train)
stable_feature_names <- feature_names[stable_indices]
stable_feature_names


################### GROUP LASSO PER TASK STABILITY SELECTION ###################
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