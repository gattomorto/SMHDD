library(glmnet)
library(caret)
library(readr)
library(gglasso)
library(SGL)
library(sparsepca)
library(reshape2)
library(leaps)
library(igraph)

rm(list=ls())

data <- read_csv("DARWIN.csv", col_names = TRUE)
data <- data[, -1]  # Remove the first column (Patient ID)

y <- as.factor(data[[ncol(data)]])  # Last column as response variable
X <- as.matrix(data[, -ncol(data)]) # All other columns as predictors
y <- as.factor(ifelse(y == "P", 1, 0))  # "P" → 1, "H" → 0

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

#X_train <- as.data.frame(X_train)
#plot(X_train$mean_jerk_on_paper20 , X_train$total_time15 )

#-------------------------------------------------------------------------------


# Find highly correlated variables (threshold = 0.8)
#high_corr_vars <- findCorrelation(correlation_matrix, cutoff = 0.8, names = TRUE)
#print(high_corr_vars)

################################ LASSO STD #####################################

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

######################### GROUP LASSO PER TASK #################################

group <- rep(1:25, each = 18)  # 450 predictors divided into 25 groups
group
y_train_numeric <- ifelse(y_train == 1, 1, -1)
y_train_numeric
y_train
X_train <- scale(X_train)  

cv_fit <- cv.gglasso(X_train, y_train_numeric, group, loss = "logit", pred.loss = "misclass", nfolds=10)

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
length(unique_numbers)


# This applies the same transformation to the test set, avoiding data leakage.
#X_test <- scale(X_test, center = attr(X_train, "scaled:center"), scale = attr(X_train, "scaled:scale"))

############################ GROUP LASSO PER FEATURE ###########################
 
y_train_numeric <- ifelse(y_train == 1, 1, -1)
ordered_indices <- order(rep(1:18, 25))
X_train <- X_train[, ordered_indices]
X_train_scaled <- scale(X_train)  
group <- rep(1:18, each = 25)

cv_fit <- cv.gglasso(X_train_scaled, y_train_numeric, group, loss = "logit", pred.loss = "misclass", nfolds=10)

plot(cv_fit)

#optimal_lambda <- cv_fit$lambda.1se
optimal_lambda <- cv_fit$lambda.1se

coefficients <- coef(cv_fit, s = optimal_lambda)
coefficients

nonzero_names <- sort(rownames(coefficients)[coefficients != 0])
nonzero_names
nonzero_names_cleaned <- gsub("\\d+$", "", nonzero_names)
nonzero_names_unique <- unique(nonzero_names_cleaned)
nonzero_names_unique

####################### SPARSE GROUP LASSO PER TASK ############################

group <- rep(1:25, each = 18)  # 450 predictors divided into 25 groups
y_train_numeric <- ifelse(y_train == 1, 1, 0)
data_list <- list(x = X_train, y = y_train_numeric)
index <- group
cv_fit <- cvSGL(data = data_list,standardize = TRUE, index = index, type = "logit", nfold = 10)
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
coefficients
feature_names <- colnames(X_train)
feature_names
all_coeff_df <- data.frame(Feature = feature_names, Coefficient = coefficients)
print(all_coeff_df)

# numero di feature diverse da 0 per ogni task
nonzero_indices <- which(coefficients != 0)
nonzero_counts <- table(factor(group[nonzero_indices], levels = unique(group)))
nonzero_counts_df <- as.data.frame(nonzero_counts)
colnames(nonzero_counts_df) <- c("Group", "Nonzero_Coefficients")
nonzero_counts_df <- nonzero_counts_df[order(nonzero_counts_df$Nonzero_Coefficients), ]
print(nonzero_counts_df)

###################### SPARSE GROUP LASSO PER FEATURE ##########################

ordered_indices <- order(rep(1:18, 25))
ordered_indices
X_train <- X_train[, ordered_indices]
group <- rep(1:18, each = 25)
y_train_numeric <- ifelse(y_train == 1, 1, 0)
data_list <- list(x = X_train, y = y_train_numeric)
index <- group
cv_fit <- cvSGL(data = data_list,standardize = TRUE, index = index, type = "logit", nfold = 10)
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

# numero di feature diverse da 0 per ogni task
nonzero_indices <- which(coefficients != 0)
nonzero_counts <- table(factor(group[nonzero_indices], levels = unique(group)))
nonzero_counts_df <- as.data.frame(nonzero_counts)
colnames(nonzero_counts_df) <- c("Group", "Nonzero_Coefficients")
nonzero_counts_df <- nonzero_counts_df[order(nonzero_counts_df$Nonzero_Coefficients), ]
print(nonzero_counts_df)







################################ SPARSE PCA ####################################

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


spca_result <- spca(X_train, k = 70, alpha = 0.009, beta=1e-10,center = TRUE, scale = TRUE)
summary(spca_result)
non_zero_loadings = get_nonzero_loadings(spca_result)
print_feature_counts(non_zero_loadings)
print_task_counts(non_zero_loadings)




########################## ELASTIC NET SOFT TRESHOLD ##########################
Ps =c()
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




############################### BEST SUBSET SELECTION ##########################

#in teoria ci riesce ma il massimo subset sizes = n
data_train <- data.frame(X_train, y_train = as.numeric(y_train) - 1)  # Convert factor to numeric 0/1
# bisogna capire se anche lasso è in realà da problemi con le variabili altamente correlate
subset_model <- regsubsets(y_train ~ ., data = data_train, nvmax = 10,really.big=T)

############################### COMPONENTS ##############################################

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

sink("components_output1.txt")
for (i in 1:components$no+1) {
  cat("Component", i-1, ":\n")
  cat(feature_names[components$membership == (i - 1)], "\n\n")
}
sink()

membership <- components$membership
names(membership) <- NULL
membership

sorted_indices <- order(membership)  # This gives the order of columns
sorted_indices
X_train_reordered <- X_train[, sorted_indices]

groups <- rep(seq_along(components$csize), times = components$csize)
groups


y_train_numeric <- ifelse(y_train == 1, 1, -1)

X_train_reordered_scaled <- scale(X_train_reordered)  

#lambda = c(0.5)
cv_fit <- cv.gglasso(X_train_reordered_scaled, y_train_numeric, groups, loss = "logit", pred.loss = "misclass", nfolds=10)

#optimal_lambda <- cv_fit$lambda.1se
optimal_lambda <- cv_fit$lambda.min

coefficients <- coef(cv_fit, s = optimal_lambda)
coefficients
coefficients <- as.vector(coefficients)

non_zero_indices <- which(coefficients[-1] != 0)
non_zero_coefficients <- coefficients[non_zero_indices + 1]
non_zero_predictors <- colnames(X_train)[non_zero_indices]
non_zero_predictors







