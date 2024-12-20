library(readr)
library(dplyr)
library(ggplot2)
library(DescTools)
library(randomForest)
library(e1071)
library(caret)
library(keras)
library(pROC)
library(rpart)
library(pheatmap)
library(vcd)

# --- LOAD DATA
dfifa <- readxl::read_excel("C:/Users/livia/Dropbox/PC/Documents/`Kuliah/Semester 5/Data Mining & Visualization/final project dashboard/oke plis fix.xlsx")
head(dfifa)
str(dfifa)
colnames(dfifa)

# --- PRE-PROCESSING

# Load Main Data with Relevant Variables
c_main <- c("value_eur", "wage_eur", "league_level", 
            "weak_foot", "skill_moves", "international_reputation", 
            "pace", "shooting", "passing", "dribbling", "defending", "physic")
main <- dfifa[c_main]
head(main)

# Missing Data Check
missing_data <- sapply(main, function(col) {
  missing_count <- sum(is.na(col))
  total_count <- length(col)
  missing_percent <- (missing_count / total_count) * 100 
  return(c(Missing = missing_count, Percent = missing_percent))
})

missing_data <- as.data.frame(t(missing_data))
print(missing_data)

# Remove Rows with Missing Values (less than 5% missing)
dfifa_clean <- dfifa %>% filter(!is.na(league_level))
main_clean <- main %>% filter(!is.na(league_level))

# Plot of Variables with Missing Values < 15%
missing_data_plot <- main_clean[c("pace", "shooting", "passing", "dribbling", 
                                  "defending", "physic")]
for (col in names(missing_data_plot)) {
  print(ggplot(main_clean, aes(x = .data[[col]])) +  # Use tidy evaluation
          geom_histogram(bins = 30, fill = "blue", color = "black", alpha = 0.7) 
          + geom_density(aes(y = ..density..), fill = "red", alpha = 0.3) +
          ggtitle(paste("Distribution of", col)) +
          theme_minimal() +
          theme(axis.text.x = element_text(angle = 45, hjust = 1)))
}
warnings() # Efek Missing Data

# Data Imputation: Replace Missing Values with the Median
main_new <- main
for (col in names(main_new)) {
  if (any(is.na(main_new[[col]]))) {
    main_new[[col]] <- ifelse(is.na(main_new[[col]]), 
                              median(main_new[[col]], na.rm = TRUE), 
                              main_new[[col]])
  }
}

# Recheck Missing Data
missing_data_check <- sapply(main_new, function(col) {
  missing_count <- sum(is.na(col))
  total_count <- length(col)
  missing_percent <- (missing_count / total_count) * 100 
  return(c(Missing = missing_count, Percent = missing_percent))
})

missing_data_check <- as.data.frame(t(missing_data_check))
print(missing_data_check)


head(main_new)

# Correlation of Numerical Variables (Spearman)
num_vars <- c("value_eur", "wage_eur", "pace", "shooting", "passing",
              "dribbling", "defending", "physic")
corr_num <- cor(main_new[, num_vars], method = "spearman")
pheatmap::pheatmap(corr_num, 
                   display_numbers = TRUE,  
                   color = colorRampPalette(c("blue", "white", "red"))(100), 
                   main = "Numerical Spearman Correlation Heatmap")

# Correlation of Categorical Variables (Cramer's V)
categ_vars <- c("weak_foot", "skill_moves", "international_reputation")
cramersv_matrix <- matrix(NA, nrow = length(categ_vars), 
                          ncol = length(categ_vars))
rownames(cramersv_matrix) <- categ_vars
colnames(cramersv_matrix) <- categ_vars

for (i in 1:length(categ_vars)) {
  for (j in i:length(categ_vars)) {
    var1 <- categ_vars[i]
    var2 <- categ_vars[j]
    if (var1 %in% colnames(main_new) && var2 %in% colnames(main_new)) {
      contingency_table <- table(main_new[[var1]], main_new[[var2]])
      cramers_v <- vcd::assocstats(contingency_table)$cramer
      cramersv_matrix[i, j] <- cramers_v
      cramersv_matrix[j, i] <- cramers_v
    }
  }
}
pheatmap::pheatmap(cramersv_matrix, 
                   display_numbers = TRUE,  
                   color = colorRampPalette(c("blue", "white", "red"))(100), 
                   main = "Categorical Cramer's V Correlation Heatmap")

# --- FEATURE SELECTION: G-Test & Kruskal-Wallis Test

target_var <- "league_level"
selected_features <- list()

# G-Test for Categorical Variables
cat("G-Test for Categorical Variables:\n")
for (cat_var in categ_vars) {
  gtest_result <- GTest(table(main_new[[cat_var]], main_new[[target_var]]))
  cat(cat_var, ": p-value =", gtest_result$p.value, "\n")
}

cat("\nSignificant Categorical Variables (p-value < 0.05):\n")
for (cat_var in categ_vars) {
  gtest_result <- GTest(table(main_new[[cat_var]], main_new[[target_var]]))
  if (gtest_result$p.value < 0.05) {
    selected_features$categorical <- c(selected_features$categorical, cat_var)
    cat(cat_var, ": p-value =", gtest_result$p.value, "\n")
  }
}

# Kruskal-Wallis Test for Numerical Variables
cat("\nKruskal-Wallis Test for Numerical Variables:\n")
for (numeric_var in num_vars) {
  kruskal_result <- kruskal.test(main_new[[numeric_var]] ~ main_new[[target_var]])
  cat(numeric_var, ": p-value =", kruskal_result$p.value, "\n")
}

cat("\nSignificant Numerical Variables (p-value < 0.05):\n")
for (numeric_var in num_vars) {
  kruskal_result <- kruskal.test(main_new[[numeric_var]] ~ main_new[[target_var]])
  if (kruskal_result$p.value < 0.05) {
    selected_features$numerical <- c(selected_features$numerical, numeric_var)
    cat(numeric_var, ": p-value =", kruskal_result$p.value, "\n")
  }
}

cat("\nSelected Features:\n")
selected_features

# --- CLASSIFICATION
# Data Balancing (Undersampling and Oversampling)

# Check class distribution
table(main_new$league_level)

# Set target sample size for each class
target_size <- 2500
balanced_data <- data.frame()

# Undersampling for Class 1
class_1 <- main_new[main_new$league_level == 1, ]
class_1_undersampled <- class_1[sample(1:nrow(class_1), target_size), ]

# Oversampling for Class 2 to 5
for (class in 2:5) {
  class_data <- main_new[main_new$league_level == class, ]
  class_oversampled <- class_data[sample(1:nrow(class_data), target_size, replace = TRUE), ]
  balanced_data <- rbind(balanced_data, class_oversampled)
}

# Combine the undersampled Class 1 data
balanced_data <- rbind(balanced_data, class_1_undersampled)
balanced_data$league_level <- factor(balanced_data$league_level)

table(balanced_data$league_level)


# -- Decision Tree & Random Forest

# Stratified Holdout (80% training, 20% testing)
set.seed(123)
trainIndex <- createDataPartition(balanced_data$league_level, p = 0.8, list = FALSE, times = 1)
train_data <- balanced_data[trainIndex, ]
test_data <- balanced_data[-trainIndex, ]

# Train Decision Tree Model
dt_model <- rpart(league_level ~ ., data = train_data, method = "class")
dt_predict <- predict(dt_model, test_data, type = "class")
dt_matrix <- confusionMatrix(dt_predict, test_data$league_level)
print(dt_matrix)
confusion_matrix_df <- as.data.frame(dt_matrix$table)

ggplot(confusion_matrix_df, aes(x = Reference, y = Prediction)) + 
  geom_tile(aes(fill = Freq), color = "white") + 
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual")
rpart.plot::rpart.plot(dt_model, type = 3, extra = 101, main = "Decision Tree Model")

# Train Random Forest Model
rf_model <- randomForest(league_level ~ ., data = train_data)
rf_predict <- predict(rf_model, test_data)
rf_matrix <- confusionMatrix(rf_predict, test_data$league_level)
print(rf_matrix)

# 5-Fold Cross-Validation 
# Ensure the factor levels of 'league_level' are valid R variable names
balanced_data$league_level <- factor(balanced_data$league_level)
levels(balanced_data$league_level) <- make.names(levels(balanced_data$league_level))

# Define 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Train Decision Tree Model using Cross-Validation
dt_cv_model <- train(league_level ~ ., data = balanced_data, method = "rpart", 
                     trControl = train_control)
print(dt_cv_model)

# Train Random Forest Model using Cross-Validation
rf_cv_model <- train(league_level ~ ., data = balanced_data, method = "rf", 
                     trControl = train_control)
print(rf_cv_model)

# Confusion Matrix for Decision Tree Cross-Validation
dt_cv_predict <- predict(dt_cv_model, balanced_data)
dt_cv_matrix <- confusionMatrix(dt_cv_predict, balanced_data$league_level)
print(dt_cv_matrix)

# Confusion Matrix for Random Forest Cross-Validation
rf_cv_predict <- predict(rf_cv_model, balanced_data)
rf_cv_matrix <- confusionMatrix(rf_cv_predict, balanced_data$league_level)
print(rf_cv_matrix)

# ROC & AUC ---- Stratified Holdout

# Train models and obtain probabilities for testing data
# Decision Tree (Probabilities)
dt_prob <- predict(dt_model, test_data, type = "prob")
str(dt_prob)

# Random Forest (Probabilities)
rf_prob <- predict(rf_model, test_data, type = "prob")

# One-vs-Rest: Calculate ROC for each class
roc_dt <- multiclass.roc(test_data$league_level, dt_prob)
roc_rf <- multiclass.roc(test_data$league_level, rf_prob)

# Calculate AUC for Decision Tree and Random Forest
auc_dt <- auc(roc_dt)
auc_rf <- auc(roc_rf)

# Print AUC for both models
print(paste("AUC for Decision Tree: ", auc_dt))
print(paste("AUC for Random Forest: ", auc_rf))

# Function to extract ROC data for each class
extract_roc_data <- function(roc_obj, model_name) {
  roc_data <- data.frame()
  
  # Loop through each class
  for (class in levels(test_data$league_level)) {
    # Create a binary response for the current class
    binary_response <- ifelse(test_data$league_level == class, 1, 0)
    
    # Calculate ROC for the current class
    roc_curve <- roc(binary_response, roc_obj$predictor[, as.numeric(class)])
    
    # Create a data frame for the current class
    class_data <- data.frame(
      FPR = 1 - roc_curve$specificities,
      TPR = roc_curve$sensitivities,
      Model = model_name,
      Class = class
    )
    
    # Combine with the main data frame
    roc_data <- rbind(roc_data, class_data)
  }
  
  return(roc_data)
}

# Extract ROC data for both models
roc_dt_data <- extract_roc_data(roc_dt, "Decision Tree")
roc_rf_data <- extract_roc_data(roc_rf, "Random Forest")

# Combine the data
roc_data <- rbind(roc_dt_data, roc_rf_data)

# Plot using ggplot2
ggplot(roc_data, aes(x = FPR, y = TPR, color = Model, linetype = Class)) +
  geom_line() +
  labs(title = "ROC Curves for Decision Tree and Random Forest using Stratified Holdout",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position = "bottom")