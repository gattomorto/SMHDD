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

#perchè ce due volta as factor?
rm(list=ls())
data <- read_csv("DARWIN.csv", col_names = TRUE)
data <- data[, -1]  # remove the first column (Patient ID)
y <- data[[ncol(data)]] 
X <- as.matrix(data[, -ncol(data)]) 
#y <- as.factor(ifelse(y == "P", 1, 0))  # "P" → 1, "H" → 0
y <- ifelse(y == "P", 1, 0) # "P" → 1, "H" → 0

set.seed(0) 

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
  
  pi_alpha <-ifelse(length(S_alpha) == 0, 0, sum(pi_x)/length(S_alpha))
  sigma_alpha = 1-length(S_alpha)/450
  phi_alpha = pi_alpha*sigma_alpha
  return(phi_alpha)
}

nlambda = 50
nalpha = 50
alphas = seq(from = 0, to = 0.003, length.out = nalpha)
#alphas = alphas[-1]
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


########################
grid_alpha <- seq(min(phis$alpha), max(phis$alpha), length.out = 100)
grid_lambda <- seq(min(phis$lambda), max(phis$lambda), length.out = 100)
grid_matrix <- expand.grid(alpha = grid_alpha, lambda = grid_lambda)




#######################################################

library(plotly)

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

library(plotly)

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


#         alpha    lambda       phi
#     0.1666667 0.9457503 0.7128748
#     0.2040816 0.7086085 0.7479630
#     0.01010101 21.86377 0.7940206 
#     0.002020202 109.3189 0.7940206 con abs2 alpha va verso lo 0 e lambda è piu grande
##################### ELASTIC NET STABILITY SELECTION ##########################
