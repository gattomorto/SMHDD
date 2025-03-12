# the column with the header "total_time8" collects the values for the "total time" feature extracted from task #8
# fare adaboost per mostrare il concetto di boosting?
# puoi fare group lasso per task?
# siamo interessati a trovare qual è il task migliore -- not all handwriting tasks are equally important for disease detection, and a targeted selection can lead to more efficient and effective diagnostic tools.
# leggere libri su feature selection?
# capire  quali task sono piu utili.
# capire quali feature sono migliori.
# sparse group lasso individua i task piu importanti e di ogni task, qual è la featura migliore

library(glmnet)
library(caret)
library(readr)
library(gglasso)
library(SGL)
library(sparsepca)

rm(list=ls())

data <- read_csv("DARWIN.csv", col_names = TRUE)
data <- data[, -1]  # Remove the first column (Patient ID)

y <- as.factor(data[[ncol(data)]])  # Last column as response variable
X <- as.matrix(data[, -ncol(data)]) # All other columns as predictors
y <- as.factor(ifelse(y == "P", 1, 0))  # "P" → 1, "H" → 0

# 4. Split Data into Training and Testing Sets (80/20)
set.seed(0) 
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

############################### PAIRWISE CORRELATION########################################

correlation_matrix <- cor(X_train)
library(reshape2)  # Load the reshape2 package for melt()
correlation_df <- melt(correlation_matrix)
correlation_df <- correlation_df[order(-abs(correlation_df$value)), ]
correlation_df <- correlation_df[correlation_df$Var1 != correlation_df$Var2, ]
print(correlation_df)

################################## LASSO STD########################################
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



spca_result <- spca(X_train, k = 70, alpha = 0.009, beta=1e-10,center = TRUE, scale = TRUE)

non_zero_loadings = get_nonzero_loadings(spca_result)
print_feature_counts(non_zero_loadings)

summary(spca_result)

summary <- summary(spca_result)
cumulatve_proportion <- summary[nrow(summary), ncol(summary)]



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





