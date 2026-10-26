
library(dplyr)
library(ggplot2)

dataset_url <- "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"

heart_data <- read.csv(dataset_url)

cat("HEART DISEASE RISK ANALYSIS\n")

cat("\n1. FIRST SIX ROWS\n")
print(head(heart_data))

cat("\n2. DATASET DIMENSIONS\n")
print(dim(heart_data))

cat("\n3. COLUMN NAMES\n")
print(names(heart_data))

cat("\n4. DATASET STRUCTURE\n")
str(heart_data)

cat("\n5. STATISTICAL SUMMARY\n")
print(summary(heart_data))

cat("\n6. MISSING VALUES\n")
print(colSums(is.na(heart_data)))

cat("\n7. DUPLICATE ROWS\n")
print(sum(duplicated(heart_data)))

heart_data <- na.omit(heart_data)
heart_data <- heart_data[!duplicated(heart_data), ]

cat("\n8. CLEANED DATASET DIMENSIONS\n")
print(dim(heart_data))

cat("\n9. AVERAGE AGE\n")
print(mean(heart_data$age))

cat("\n10. AVERAGE CHOLESTEROL\n")
print(mean(heart_data$chol))

cat("\n11. AVERAGE BLOOD PRESSURE\n")
print(mean(heart_data$trestbps))

cat("\n12. AVERAGE MAXIMUM HEART RATE\n")
print(mean(heart_data$thalach))

cat("\n13. TARGET COUNTS\n")
print(table(heart_data$target))

cat("\n14. AVERAGE AGE BY TARGET\n")
print(
  heart_data %>%
    group_by(target) %>%
    summarise(Average_Age = mean(age))
)

cat("\n15. AVERAGE CHOLESTEROL BY TARGET\n")
print(
  heart_data %>%
    group_by(target) %>%
    summarise(Average_Cholesterol = mean(chol))
)

print(
  ggplot(heart_data, aes(x = age)) +
    geom_histogram(bins = 15, fill = "skyblue", color = "black") +
    labs(title = "Age Distribution", x = "Age", y = "Frequency") +
    theme_minimal()
)

print(
  ggplot(heart_data, aes(x = chol)) +
    geom_histogram(bins = 15, fill = "orange", color = "black") +
    labs(title = "Cholesterol Distribution", x = "Cholesterol", y = "Frequency") +
    theme_minimal()
)

print(
  ggplot(heart_data, aes(x = factor(target), y = age)) +
    geom_boxplot(fill = "lightgreen") +
    labs(title = "Age by Heart Disease Target", x = "Target", y = "Age") +
    theme_minimal()
)

print(
  ggplot(heart_data, aes(x = trestbps, y = chol, color = factor(target))) +
    geom_point(size = 2) +
    labs(title = "Blood Pressure vs Cholesterol",
         x = "Resting Blood Pressure",
         y = "Cholesterol",
         color = "Target") +
    theme_minimal()
)

print(
  ggplot(heart_data, aes(x = factor(target), fill = factor(target))) +
    geom_bar() +
    labs(title = "Heart Disease Target Distribution",
         x = "Target",
         y = "Number of People",
         fill = "Target") +
    theme_minimal()
)

heart_counts <- as.data.frame(table(heart_data$target))
colnames(heart_counts) <- c("Target", "Count")

print(
  ggplot(heart_counts, aes(x = "", y = Count, fill = Target)) +
    geom_col(width = 1) +
    coord_polar(theta = "y") +
    labs(title = "Heart Disease Distribution - Pie Chart",
         fill = "Target") +
    theme_void()
)

print(
  ggplot(heart_data, aes(x = age, y = chol)) +
    geom_line(stat = "summary", fun = mean, color = "blue") +
    labs(title = "Age vs Average Cholesterol",
         x = "Age",
         y = "Average Cholesterol") +
    theme_minimal()
)

print(
  ggplot(heart_data, aes(x = factor(sex), fill = factor(target))) +
    geom_bar(position = "dodge") +
    labs(title = "Heart Disease by Sex",
         x = "Sex (0 = Female, 1 = Male)",
         y = "Number of People",
         fill = "Target") +
    theme_minimal()
)

print(
  ggplot(heart_data, aes(x = thalach)) +
    geom_histogram(bins = 15, fill = "purple", color = "black") +
    labs(title = "Maximum Heart Rate Distribution",
         x = "Maximum Heart Rate",
         y = "Frequency") +
    theme_minimal()
)

print(
  ggplot(heart_data, aes(x = age, y = thalach, color = factor(target))) +
    geom_point(size = 2) +
    labs(title = "Age vs Maximum Heart Rate",
         x = "Age",
         y = "Maximum Heart Rate",
         color = "Target") +
    theme_minimal()
)

set.seed(123)

heart_data$target <- factor(heart_data$target, levels = c(0, 1))

train_rows <- sample(
  seq_len(nrow(heart_data)),
  size = floor(0.8 * nrow(heart_data))
)

train_data <- heart_data[train_rows, ]
test_data <- heart_data[-train_rows, ]

heart_model <- glm(
  target ~ age + sex + cp + trestbps + chol +
    fbs + restecg + thalach + exang + oldpeak +
    slope + ca + thal,
  data = train_data,
  family = binomial
)

cat("\n16. LOGISTIC REGRESSION MODEL\n")
print(summary(heart_model))

predicted_probabilities <- predict(
  heart_model,
  newdata = test_data,
  type = "response"
)

predicted_classes <- ifelse(predicted_probabilities >= 0.5, 1, 0)

predicted_classes <- factor(
  predicted_classes,
  levels = c(0, 1)
)

cat("\n17. CONFUSION MATRIX\n")
print(
  table(
    Actual = test_data$target,
    Predicted = predicted_classes
  )
)

accuracy <- mean(predicted_classes == test_data$target) * 100

cat("\n18. MODEL ACCURACY (%)\n")
print(round(accuracy, 2))

cat("\nPROJECT COMPLETED SUCCESSFULLY!\n")
