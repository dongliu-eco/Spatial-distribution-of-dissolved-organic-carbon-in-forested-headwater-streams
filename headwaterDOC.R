# Load necessary libraries
library(randomForest)
library(reshape2)
library(ggplot2)
library(ranger)
library(DALEX)
library(dplyr)

# Load dataset
stream <- read.csv("D:/2024-2025/headwater/train_data.csv")

# Split the dataset into training (80%) and testing (20%) sets
set.seed(123)
train <- sample(nrow(stream), nrow(stream) * 0.8)
stream_train <- stream[train, ]
stream_test <- stream[-train, ]

# Define parameter ranges for grid search
ntree_values <- seq(100, 1000, by = 100)
mtry_values <- c(1, 3, 5, 7)
num.cores_values <- c(1, 2, 4)

# Initialize best score and best parameter combination
best_score <- Inf
best_parameters <- NULL

# Grid search for hyperparameter tuning
for (ntree in ntree_values) {
  for (mtry in mtry_values) {
    for (num.cores in num.cores_values) {
      # Train random forest model
      rf_model <- randomForest(DOC ~ ., data = stream_train, ntree = ntree, mtry = mtry, num.cores = num.cores)
      
      # Cross-validation to compute error score
      scores <- rfcv(stream_train[-ncol(stream_train)], stream_train$DOC, cv.fold = 10, step = 1.5)
      
      # Update best parameters if current model performs better
      if (all(scores$error.cv < best_score)) {
        best_score <- min(scores$error.cv)
        best_parameters <- list(ntree = ntree, mtry = mtry, num.cores = num.cores)
      }
    }
  }
}

# Print best parameter combination
print(best_parameters)

# Cross-validation with 5 repetitions of 10-fold CV
set.seed(123)
stream_train.cv <- replicate(5, rfcv(stream_train[-ncol(stream_train)], stream_train$DOC, cv.fold = 10, step = 1.5), simplify = FALSE)

# Extract cross-validation results and reshape for visualization
stream_train.cv <- data.frame(sapply(stream_train.cv, '[[', 'error.cv'))
stream_train.cv$DOC <- rownames(stream_train.cv)
stream_train.cv <- melt(stream_train.cv, id = 'DOC')
stream_train.cv$DOC <- as.numeric(as.character(stream_train.cv$DOC))
stream_train.cv.mean <- aggregate(stream_train.cv$value, by = list(stream_train.cv$DOC), FUN = mean)

# Plot cross-validation error
ggplot(stream_train.cv.mean, aes(Group.1, x)) +
  geom_line() +
  theme(panel.grid = element_blank(), panel.background = element_rect(color = 'black', fill = 'transparent')) +
  labs(x = 'Number of variables', y = 'Cross-validation error')

# Fine-tuning mtry parameter
tuneRF(stream_train[, -1], stream_train[, 1], stepFactor = 1.5)

# Prediction on the testing dataset
stream_train.forest <- randomForest(DOC ~ ., data = stream_train, ntree = best_parameters$ntree, mtry = best_parameters$mtry, num.cores = best_parameters$num.cores)
predict_forest <- predict(stream_train.forest, stream_test)

# Calculate Mean Squared Error (MSE)
mse <- mean((predict_forest - stream_test$DOC)^2)
print(mse)

# Uncertainty estimation using infinitesimal Jackknife method
set.seed(123)
model_fit <- ranger(DOC ~ ., 
                    data = stream_train, 
                    importance = 'impurity', 
                    num.trees = best_parameters$ntree, 
                    mtry = best_parameters$mtry, 
                    num.threads = best_parameters$num.cores, 
                    keep.inbag = TRUE) # Keep inbag parameter for variance estimation

# Predict with standard deviation estimation
model_fit.predictions <- predict(model_fit, data = stream_test, type = "se", se.method = "infjack")

# Extract standard deviation of predictions
sd_of_model_fit <- model_fit.predictions$se

# Add standard deviation to the testing dataset
stream_test$prediction_sd <- sd_of_model_fit

# Save the dataset with uncertainty information
write.csv(stream_test, "D:/2024-2025/headwater/stream_test_with_sd.csv", row.names = FALSE)

# Summary of uncertainty
head(stream_test)

# Construct model explainer
rf_exp <- DALEX::explain(stream_train.forest,
                         data = stream_train[,-1],
                         y = stream_train$DOC,
                         label = "randomForest")

# Partial dependence profile
pdp_rf <- model_profile(rf_exp)

# Prepare data for plotting
pdp_data_plot <- pdp_rf$agr_profiles %>% 
  dplyr::select(Variable = `_vname_`, x = `_x_`, y_hat = `_yhat_`)

# Plot partial dependence profiles
pdp_plot <- pdp_data_plot %>% 
  ggplot(aes(x, y_hat))+
  facet_wrap(~Variable, scales = "free_y", nrow = 3, ncol = 4)+
  ylab("Normalized DOC concentration")+
  geom_smooth(method = "loess", se = T, fill = "blue", color = "black", alpha = 0.4, linewidth = 1.2)+
  theme(panel.border = element_rect(color = "black", linewidth = 1, fill = NA),
        panel.background = element_rect(fill = "white"),
        axis.text.x = element_text(color = "black", size = 14),
        axis.text.y = element_text(color = "black", size = 14),
        axis.title.y = element_text(color = "black", size = 16),
        strip.text.x = element_text(color = "black", size = 16))

print(pdp_plot)
