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

y <- as.factor(data[[ncol(data)]])  # Last column as response variable
X <- as.matrix(data[, -ncol(data)]) # All other columns as predictors
y <- as.factor(ifelse(y == "P", 1, 0))  # "P" → 1, "H" → 0
X <- scale(X)

set.seed(0) 
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

############################ PAIRWISE CORRELATION ##############################

correlation_matrix <- cor(X_train)
# questo blocco rimuove tutte le rindondanze
correlation_df <- melt(correlation_matrix)
correlation_df <- correlation_df[order(-abs(correlation_df$value)), ]
correlation_df <- correlation_df[correlation_df$Var1 != correlation_df$Var2, ]
correlation_df$pair <- apply(correlation_df[, c("Var1", "Var2")], 1, function(x) paste(sort(x), collapse = "_"))
correlation_df <- correlation_df[!duplicated(correlation_df$pair), ]
correlation_df$pair <- NULL
correlation_df$abs_value = abs(correlation_df$value)
correlation_df$abs_value <- ifelse(correlation_df$abs_value > 0.8, 
                                            correlation_df$abs_value, 
                                            0)

X_train <- as.data.frame(X_train)
plot(X_train$max_x_extension1 , X_train$max_y_extension1 )
cor(X_train$max_x_extension1, X_train$max_y_extension1)



# Find highly correlated variables (threshold = 0.8)
#high_corr_vars <- findCorrelation(correlation_matrix, cutoff = 0.8, names = TRUE)
#print(high_corr_vars)

################################ LASSO STD (to be removed) #####################################

# cv_model$lambda: è la sequenza di lambda testati (100)
# cv_model$lambda.min
# cv_model$lamda.1se
# cv_model$cvsd: standard error per ogni lambda
# cv_model$nzero: Number of nonzero coefficients for each lambda
# glmnet.fit: contine i coefficienti per ogni lambda addestrato su tutti il dataset. cv_model$glmnet.fit$df: il numero di coefficienti usati
cv_model <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1, nfolds = 10)
best_lambda <- cv_model$lambda.min  
best_lambda <- cv_model$lambda.1se 

# Train Final Lasso Model with Best Lambda
lasso_model <- glmnet(X_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)

# Make Predictions
probabilities <- predict(lasso_model, X_test, type = "response")
#length(y_test): 34
probabilities
predictions <- ifelse(probabilities > 0.5, 1, 0)  
# Evaluate Performance
conf_matrix <- confusionMatrix(as.factor(predictions), y_test)
print(conf_matrix)


coefficients <- coef(lasso_model)
# Convert the coefficients to a vector and remove the intercept
coefficients <- as.vector(coefficients)[-1]
non_zero_indices <- which(coefficients != 0)
non_zero_column_names <- colnames(X_train)[non_zero_indices]
non_zero_column_names_cleaned <- sub("\\d+$", "", non_zero_column_names)
frequency_table <- table(non_zero_column_names_cleaned)
sorted_frequency_table <- sort(frequency_table, decreasing = TRUE)
print(sorted_frequency_table)

######################### GROUP LASSO PER TASK (to be removed) #################################

group <- rep(1:25, each = 18)  # 450 predictors divided into 25 groups
group
y_train_numeric <- ifelse(y_train == 1, 1, -1)
y_train_numeric
y_train
X_train <- scale(X_train)  

cv_fit <- cv.gglasso(X_train, y_train_numeric, group, loss = "logit", pred.loss = "misclass", nfolds=10)
plot.gglasso(cv_fit)
# trovo i gruppi selezionati
optimal_lambda <- cv_fit$lambda.1se
#optimal_lambda <- cv_fit$lambda.min
coefficients <- coef(cv_fit, s = optimal_lambda)
coefficients
coefficients <- as.vector(coefficients)
non_zero_indices <- which(coefficients[-1] != 0)
non_zero_coefficients <- coefficients[non_zero_indices + 1]
non_zero_predictors <- colnames(X_train)[non_zero_indices]
non_zero_predictors
numbers <- as.numeric(gsub("[^0-9]", "", non_zero_predictors))
unique_numbers <- unique(numbers)
unique_numbers


# This applies the same transformation to the test set, avoiding data leakage.
#X_test <- scale(X_test, center = attr(X_train, "scaled:center"), scale = attr(X_train, "scaled:scale"))

############################ GROUP LASSO PER FEATURE (to be removed) ###########################
 
y_train_numeric <- ifelse(y_train == 1, 1, -1)
ordered_indices <- order(rep(1:18, 25))
X_train <- X_train[, ordered_indices]
X_train_scaled <- scale(X_train)  
group <- rep(1:18, each = 25)

cv_fit <- cv.gglasso(X_train_scaled, y_train_numeric, group, loss = "logit", pred.loss = "misclass", nfolds=10)

# trovo i gruppi selezionati
#optimal_lambda <- cv_fit$lambda.1se
optimal_lambda <- cv_fit$lambda.1se
coefficients <- coef(cv_fit, s = optimal_lambda)
coefficients
nonzero_names <- sort(rownames(coefficients)[coefficients != 0])
nonzero_names
nonzero_names_cleaned <- gsub("\\d+$", "", nonzero_names)
nonzero_names_unique <- unique(nonzero_names_cleaned)
nonzero_names_unique

####################### SPARSE GROUP LASSO PER TASK (to be removed) ############################

group <- rep(1:25, each = 18)  # 450 predictors divided into 25 groups
y_train_numeric <- ifelse(y_train == 1, 1, 0)
data_list <- list(x = X_train, y = y_train_numeric)
index <- group
cv_fit <- cvSGL(data = data_list, standardize = TRUE, index = index, type = "logit", nfold = 3)
plot(cv_fit)

#questo blocco trova lambda.min e lambda.1se
cv_errors <- cv_fit$lldiff
cv_sd <- cv_fit$llSD
lambda.min_index <- which.min(cv_errors)
lambda.min <- cv_fit$lambdas[lambda.min_index]
lambda.min
abline(v=log(lambda.min), col="red",lty=2)
min_loss <- cv_errors[lambda.min_index]
threshold <- min_loss + cv_sd[lambda.min_index]
threshold
abline(h=threshold, col="grey")
# Find the most regularized (largest lambda) within 1SE range
lambda.1se_index <- min(which(cv_errors <= threshold))
lambda.1se_index
lambda.1se <- cv_fit$lambdas[lambda.1se_index]
lambda.1se
abline(v=log(lambda.1se), col="grey")
lambda.1se_index
lambda.min_index

#tutte le feature con i beta
coefficients <- cv_fit$fit$beta[, lambda.1se_index]
coefficients
feature_names <- colnames(X_train)
feature_names
all_coeff_df <- data.frame(Feature = feature_names, Coefficient = coefficients)
print(all_coeff_df)

# numero di feature diverse da 0 per ogni task -- task 9 sembra piu importante perchè ha molte feature attive
nonzero_indices <- which(coefficients != 0)
nonzero_counts <- table(factor(group[nonzero_indices], levels = unique(group)))
nonzero_counts_df <- as.data.frame(nonzero_counts)
colnames(nonzero_counts_df) <- c("Group", "Nonzero_Coefficients")
nonzero_counts_df <- nonzero_counts_df[order(nonzero_counts_df$Nonzero_Coefficients), ]
print(nonzero_counts_df)

###################### SPARSE GROUP LASSO PER FEATURE (to be removed) ##########################

ordered_indices <- order(rep(1:18, 25))
ordered_indices
X_train <- X_train[, ordered_indices]
group <- rep(1:18, each = 25)
y_train_numeric <- ifelse(y_train == 1, 1, 0)
data_list <- list(x = X_train, y = y_train_numeric)
index <- group
cv_fit <- cvSGL(data = data_list,standardize = TRUE, index = index, type = "logit", nfold = 3)
plot(cv_fit)

#questo blocco trova lambda.min e lambda.1se
cv_errors <- cv_fit$lldiff
cv_sd <- cv_fit$llSD
lambda.min_index <- which.min(cv_errors)
lambda.min <- cv_fit$lambdas[lambda.min_index]
lambda.min
abline(v=log(lambda.min), col="red",lty=2)
min_loss <- cv_errors[lambda.min_index]
threshold <- min_loss + cv_sd[lambda.min_index]
threshold
abline(h=threshold, col="grey")
# Find the most regularized (largest lambda) within 1SE range
lambda.1se_index <- min(which(cv_errors <= threshold))
lambda.1se_index
lambda.1se <- cv_fit$lambdas[lambda.1se_index]
lambda.1se
abline(v=log(lambda.1se), col="grey")
lambda.1se_index
lambda.min_index

#tutti coefficienti con i beta
coefficients <- cv_fit$fit$beta[, lambda.1se_index]
feature_names <- colnames(X_train)
all_coeff_df <- data.frame(Feature = feature_names, Coefficient = coefficients)
print(all_coeff_df)

# per ogni feature mostro i task in cui è usata
nonzero_indices <- which(coefficients != 0)
nonzero_counts <- table(factor(group[nonzero_indices], levels = unique(group)))
nonzero_counts_df <- as.data.frame(nonzero_counts)
colnames(nonzero_counts_df) <- c("Group", "Nonzero_Coefficients")
nonzero_counts_df <- nonzero_counts_df[order(nonzero_counts_df$Nonzero_Coefficients), ]
print(nonzero_counts_df)

################################ SPARSE PCA (to be removed) ####################################

# ottengo le variabili che partecipano nella formazione dei PC
get_nonzero_loadings <- function(spca_result) 
{
  loadings <- spca_result$loadings  
  variable_names <- colnames(X_train)
  loadings_df <- as.data.frame(loadings)
  rownames(loadings_df) <- variable_names 
  non_zero_loadings <- loadings_df[rowSums(loadings_df != 0) > 0, , drop = FALSE]
  non_zero_loadings_abs <- abs(non_zero_loadings)
  #print(paste("numero di variabili non zero:", nrow(non_zero_loadings)))
  #return(nrow(non_zero_loadings))
  return(non_zero_loadings)
}

# per ogni feature calcolo quante volte compare
print_feature_counts <- function(non_zero_loadings) 
{
  row_names <- rownames(non_zero_loadings)
  cleaned_names <- gsub("\\d+$", "", row_names)
  name_counts <- table(cleaned_names)
  sorted_name_counts <- sort(name_counts, decreasing = TRUE)
  print(sorted_name_counts)
  print(paste("numero di feature uniche che compaiono:",length(unique(cleaned_names)),"/ 18"))
  
}

# per ogni task calcolo quante volte compare
print_task_counts <- function(non_zero_loadings) 
{
  row_names <- rownames(non_zero_loadings)
  modified_names <- sub(".*?(\\d+)$", "task\\1", row_names)
  count_table <- table(modified_names)
  sorted_table <- sort(count_table, decreasing = TRUE)
  print(sorted_table)
  
}


# meglio chiamarla optimize_phi e ritornare k_max, alpha_max
evalute_phi <- function() 
{
  results <- data.frame(k = integer(),alpha = numeric(),num_non_zero_loadings = integer(),prop_var_explained = numeric(),phi = numeric(),stringsAsFactors = FALSE)
  # migliore: k = 70, alpha = 0.009, phi = 0.2977333
  ks = c(10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,150,150,180, 190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450)
  alphas = c(0.05, 0.045, 0.04,0.035,0.03,0.025,0.02,0.015,0.01,0.009,0.008,0.007, 0.006, 0.005, 0.004, 0.003)
  ks = c(1,2)
  alphas = c(0.05,0.04)
  
  for (k in ks) 
  {
    for (alpha in alphas) 
    {
      cat("Current k:", k, "| Current alpha:", alpha, "\n")
      spca_result <- spca(X_train, k = k, alpha = alpha, beta=1e-10,center = TRUE, scale = TRUE, verbose = FALSE)
      
      num_nzl = nrow(get_nonzero_loadings(spca_result))
      prop_variables_used = num_nzl/450
      
      summary <- summary(spca_result)
      prop_var_explained <- summary[nrow(summary), ncol(summary)]
      
      # numero di variabili selezionate vs varianza spiegata
      phi = (1-prop_variables_used)*prop_var_explained
      
      results <- rbind(results, list(k = k, alpha = alpha, num_non_zero_loadings = num_nzl, prop_var_explained = prop_var_explained, phi = phi))
      
    }
  }
  return(results)
}


vv = evalute_phi()


spca_result <- spca(X_train, k = 70, alpha = 0.009, beta=1e-10,center = TRUE, scale = TRUE, verbose = FALSE)
summary(spca_result)
non_zero_loadings = get_nonzero_loadings(spca_result)
print_feature_counts(non_zero_loadings)
print_task_counts(non_zero_loadings)

)=()


########################## GROUP LASSO ON COMPONENTS (to be removed) ###########################

# corr_matrix <- cor(X_train)
# graph <- graph_from_adjacency_matrix(abs(corr_matrix) > 0.8, mode = "undirected")
# cliques <- max_cliques(graph)
# sink("components_output_u.txt")
# #print(cliques)
# sink()

# estrai componenti
corr_matrix <- cor(X_train)
adj_matrix <- abs(corr_matrix) > 0.8
g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", diag = FALSE)
components <- components(g)
feature_names <- colnames(X_train)

# stampa le componenti
#sink("components_output1.txt")
for (i in 1:components$no+1) {
  cat("Component", i-1, ":\n")
  cat(feature_names[components$membership == (i - 1)], "\n\n")
}
#sink()

# prepara i gruppi (provare a fare nella versione piu semplificata senza permutare X)
membership <- components$membership
names(membership) <- NULL
membership
sorted_indices <- order(membership)  # This gives the order of columns
sorted_indices
X_train_reordered <- X_train[, sorted_indices] # cambia le colonne, non righe, percui non bisogna cambiare y
groups <- rep(seq_along(components$csize), times = components$csize)
groups

# train 
y_train_numeric <- ifelse(y_train == 1, 1, -1)
X_train_reordered_scaled <- scale(X_train_reordered)  
cv_fit <- cv.gglasso(X_train_reordered_scaled, y_train_numeric, groups, loss = "logit", pred.loss = "misclass", nfolds=10)
optimal_lambda <- cv_fit$lambda.min

# chosen coefficients
coefficients <- coef(cv_fit, s = optimal_lambda)
coefficients
coefficients <- as.vector(coefficients)
non_zero_indices <- which(coefficients[-1] != 0)
non_zero_coefficients <- coefficients[non_zero_indices + 1]
non_zero_predictors <- colnames(X_train_reordered)[non_zero_indices]
non_zero_predictors

# controlla la correttezza
check <- function(beta_lam, num_groups) {
  for (gr in 1:num_groups) 
  {
    beta_gr <- beta_lam[groups == gr]
    if (any(beta_gr == 0.0) ) 
    {
      if(any(beta_gr!=0.0))
      {
        print("err")
      }
    }
    
    if (any(beta_gr != 0.0) ) 
    {
      if(any(beta_gr==0.0))
      {
        print("err")
      }
      
    }
    
  }
  print("tutto ok")
}
check(coefficients[-1], length(unique(groups)))

# stampa i gruppi selezionati
for (gr in 1:length(unique(groups))) 
{
  # tutti i coefficienti del gruppo g 
  beta_gr <- coefficients[-1][groups == gr]
  if (any(beta_gr != 0)) 
  {
    print(gr)
  }
}



############################### BEST SUBSET SELECTION (to be removed) ##########################

#in teoria ci riesce ma il massimo subset sizes = n
data_train <- data.frame(X_train, y_train = as.numeric(y_train) - 1)  # Convert factor to numeric 0/1
# bisogna capire se anche lasso è in realà da problemi con le variabili altamente correlate
subset_model <- regsubsets(y_train ~ ., data = data_train, nvmax = 10,really.big=T)


################################ CLUSTER ANALYSIS ######################################

cor_matrix <- cor(X_train)
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









######################### ALPHA SELECTION ELASTIC NET ##########################
Ps = c()
alphas = seq(from = 0, to = 1, length.out = 1000)
for (alpha in alphas) 
{
  cv_fit <- cv.glmnet(X_train, y_train, alpha = alpha, family = "binomial")
  best_lambda <- cv_fit$lambda.min
  elastic_net_model <- glmnet(X_train, y_train, alpha = alpha, lambda = best_lambda, family = "binomial")
  
  # estraggo i coefficienti diversi da 0 (nonzero_vars)
  coef_enet <- coef(elastic_net_model)
  coef_values <- as.vector(coef_enet[-1])  
  names(coef_values) <- rownames(coef_enet)[-1] 
  nonzero_vars <- names(coef_values[coef_values != 0])
  #nonzero_vars
  #nonzero_vars = c("num_of_pendown5","paper_time25","air_time24")
  #nonzero_vars = c()

  correlation_df_enet_boundary <- correlation_df[xor(correlation_df$Var1 %in% nonzero_vars, correlation_df$Var2 %in% nonzero_vars), ]
  # correlazione media mancante
  P1 <- ifelse(nrow(correlation_df_enet_boundary) == 0, 0, sum(correlation_df_enet_boundary$abs_value) / nrow(correlation_df_enet_boundary))
  correlation_df_enet_interior <- correlation_df[(correlation_df$Var1 %in% nonzero_vars) & (correlation_df$Var2 %in% nonzero_vars), ]
  # correlazione media presente
  P2 <- ifelse(nrow(correlation_df_enet_interior) == 0, 1, sum(correlation_df_enet_interior$abs_value) / nrow(correlation_df_enet_interior))
  P = (1-P1)*P2
  Ps=c(Ps,P)
  
  cat("alpha: ",alpha," P: ",P,"\n")
}

loess_fit <- loess(Ps ~ alphas, span = 0.3)  
Ps_smooth <- predict(loess_fit, newdata = alphas)
max_index <- which.max(Ps_smooth)
alpha_max <- alphas[max_index]
alpha_max
P_max <- Ps_smooth[max_index]
plot(alphas, Ps, pch = 16, col = "gray")
lines(alphas, Ps_smooth, lwd = 2)

########################## ELASTIC NET STABILITY SELECTION #####################
alpha = 0.12
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
    
    coefficients <- coef(subsample_model, s = lambda)[-1]  # Exclude intercept
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


################### GROUP LASSO PER TASK STABILITY SELECTION #############################
groups <- rep(1:25, each = 18) 
y_train <- ifelse(y_train == 1, 1, 0)
data <- list(x = X_train, y = y_train)
#lambdas <- seq(from = 0.0129, to = 0.01, length.out = 20)#sgl alpha = 0
#nlam = length(lambdas)
#lambdas
SGL_model <- SGL(data,groups, type="logit",standardize = TRUE, verbose = TRUE, alpha = 0)
lambdas <- SGL_model$lambdas
#lambdas = lambdas[0:3]
lambdas
# non si usa
beta_matrix <- SGL_model$beta 

num_groups <- 25
selection_frequencies <- matrix(0, nrow =length(lambdas) , ncol =num_groups )
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
    # subsample_data?
    data <- list(x = X_subsample, y = y_subsample)
    subsample_model <- SGL(data,groups, type="logit",lambdas = lambda,nlam = 1,standardize = TRUE, verbose = FALSE, alpha = 0)
    beta <- subsample_model$beta
    
    # per ogni gruppo controllo se almeno un coefficiente è non nullo
    for (gr in 1:25) 
    {
      # tutti i coefficienti del gruppo g (lunghezza 18)
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
stable_groups <- which(max_selection_probabilities >= 0.8)
stable_groups


################## GROUP LASSO PER FEATURE STABILITY SELECTION ###########################

#dati i coefficienti di un modello con un certo lambda, controlla che i coefficienti dei gruppi che siano o tutti zero o tutti non zero
check <- function(beta_lam) {
  for (gr in 1:18) 
  {
    beta_gr <- beta_lam[groups == gr]
    if (any(beta_gr == 0.0) ) 
    {
      if(any(beta_gr!=0.0))
      {
        print("err")
      }
    }
    
    if (any(beta_gr != 0.0) ) 
    {
      if(any(beta_gr==0.0))
      {
        print("err")
      }
      
    }
    
  }
}

groups <- rep(1:18, times = 25)
y_train <- ifelse(y_train == 1, 1, 0)
data <- list(x = X_train, y = y_train)
SGL_model <- SGL(data,groups, type="logit", nlam = 5, standardize = TRUE, verbose = TRUE, alpha = 0)
lambdas <- SGL_model$lambdas
lambdas
# questa non si usa
beta_matrix <- SGL_model$beta 

num_features <- 18
selection_frequencies <- matrix(0, nrow =length(lambdas) , ncol =num_features )
num_subsamples <- 50
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
    # subsample_data?
    data <- list(x = X_subsample, y = y_subsample)
    subsample_model <- SGL(data,groups, type="logit",lambdas = lambda,nlam = 1,standardize = TRUE, verbose = FALSE, alpha = 0)
    beta <- subsample_model$beta
    check(beta)
    # considero solo i primi 18 valori perchè ognuno è rappresentante di ogni gruppo
    beta1 = beta[0:18]
    #print(beta1)
    # i coefficienti diversi da 0 sono selezionati
    beta1[beta1 != 0] <- 1
    #print(beta1)
    selection_frequencies[lambda_idx,] <- selection_frequencies[lambda_idx,] + beta1
  }
}
selection_probabilities <- selection_frequencies / num_subsamples
max_selection_probabilities <- apply(selection_probabilities, 2, max)
max_selection_probabilities
stable_features <- which(max_selection_probabilities >= 0.8)
stable_features
################ GROUP LASSO ON COMPONENTS STABILITY SELECTION #################

check <- function(beta_lam, num_groups) {
  for (gr in 1:num_groups) 
  {
    beta_gr <- beta_lam[groups == gr]
    if (any(beta_gr == 0.0) ) 
    {
      if(any(beta_gr!=0.0))
      {
        print("err")
      }
    }
    
    if (any(beta_gr != 0.0) ) 
    {
      if(any(beta_gr==0.0))
      {
        print("err")
      }
      
    }
    
  }
  print("tutto ok")
}

# estrai componenti
corr_matrix <- cor(X_train)
adj_matrix <- abs(corr_matrix) > 0.7
g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", diag = FALSE)
components <- components(g)
feature_names <- colnames(X_train)

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
y_train <- ifelse(y_train == 1, 1, 0)
data <- list(x = X_train, y = y_train)
SGL_model <- SGL(data, groups, type = "logit", nlam = 5, standardize = TRUE, verbose = TRUE, alpha = 0)
lambdas <- SGL_model$lambdas
lambdas
print(SGL_model)
# e l'intercept dov'è?
beta_matrix <- SGL_model$beta 
num_groups = length(unique(groups))
num_groups
selection_frequencies <- matrix(0, nrow =length(lambdas) , ncol = num_groups )
num_subsamples <- 10
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
    # subsample_data?
    data <- list(x = X_subsample, y = y_subsample)
    subsample_model <- SGL(data,groups, type="logit",lambdas = lambda,nlam = 1,standardize = TRUE, verbose = FALSE, alpha = 0)
    beta <- subsample_model$beta
    #check(beta, num_groups)
    
    # per ogni gruppo controllo se almeno un coefficiente è non nullo
    for (gr in 1:num_groups) 
    {
      # tutti i coefficienti del gruppo g 
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
stable_groups <- which(max_selection_probabilities >= 0.95)
stable_groups
for (stable_group in stable_groups) {
  cat("Stable Group", stable_group, ":\n")
  features_in_group <- feature_names[groups == stable_group]
  cat(features_in_group, "\n\n")
}

###################### TASK GROUP BEST SUBSET SELECTION ########################

gbss.fit <- function(X, y, s, metric = "dev") 
{
  ##num_groups (s) = 2
  numbers <- 1:25
  combinations <- t(combn(numbers, s))
  accuracy <- c()
  deviance <- c()
  #models <- list()
  
  groups <- rep(1:25, each = 18) 
  num_combinations <- nrow(combinations)
  num_combinations
  
  best_model = NULL
  best_dev = Inf
  
  for (i in 1:num_combinations) 
  {
    print(i)
    xx <- combinations[i, ]
    feature_selector <- groups %in% xx
    X_selected <- X[, feature_selector, drop = FALSE]
    model <- glm(y ~ ., data = data.frame(y, X_selected), family = binomial, control = glm.control(maxit = 100))
    #models[[i]] <- model
    deviance[i] <-  model$deviance
    prob_predictions <- predict(model, newdata = data.frame(X_selected), type = "response")
    y_pred <- ifelse(prob_predictions > 0.5, 1, 0)
    accuracy[i] <-  mean(y_pred == y)
    
    if (model$deviance < best_dev)
    {
      best_dev = model$deviance
      best_model = model
    }
    
  }
  
  combinations <- cbind(combinations, accuracy)
  combinations <- cbind(combinations, deviance)
  
  #best_combination <- combinations[which.min(combinations[,"deviance"]), 1:(ncol(combinations) - 2)]
  #return(best_combination)
  return(best_model)
  
}
#zz = ff(X_train,y_train,2)
#16 25
#1  9 25

cv.gbss <- function(X,y,S,nfolds=3) 
{ 
  cvm <- numeric(length(S))
  names(cvm) <- S 
  
  for (i in seq_along(S))
  {
    s <- S[i]
    folds <- sample(rep(1:nfolds, length.out = nrow(X))) 
    misclassification_rates <- numeric(nfolds)
    for (k in 1:nfolds) 
    {
      X_train <- X[folds != k, ]
      X_test <- X[folds == k, ]
      y_train <- y[folds != k]
      y_test <- y[folds == k]
      
      model <- gbss.fit(X_train,y_train,s)
      # attenzione che qui newdata ha numero di colonne diverso
      prob_predictions <- predict(model, newdata = data.frame(X_test), type = "response")
      y_pred <- ifelse(prob_predictions > 0.5, 1, 0)
      misclassification_rates[k] <- mean(y_pred != y_test)
    }
    
    cvm[i] <- mean(misclassification_rates)
  }
  
  s.min <- S[which.min(cvm)]
  return(list(cvm = cvm, s.min = s.min))
  
}

xx = cv.gbss(X,y,c(4))
