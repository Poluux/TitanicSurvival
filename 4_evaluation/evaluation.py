import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# This file evaluates the performance and precision of our model

# Load model and test data
model = joblib.load("../3_regression/logistic_model.joblib")
X_test = np.load("../3_regression/X_test.npy")
y_test = np.load("../3_regression/y_test.npy")

# Predict on the entire test set
y_pred = model.predict(X_test)

# Function to compute and print metrics given true and predicted labels
def print_metrics(y_true, y_pred, group_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    
    print(f"=== Evaluation for {group_name} ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    
    print(f"--- Classification report for {group_name} ---")
    print(classification_report(y_true, y_pred))
    
    plt.figure(figsize=(5,4))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Survived', 'Survived'], 
                yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Prediction')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for {group_name}')
    plt.show()

# Evaluate on whole test set
print_metrics(y_test, y_pred, "All passengers")

# Split test data by Sex: 0 = male, 1 = female
male_indices = X_test[:, 0] == 0
female_indices = X_test[:, 0] == 1

# Predictions for males and females
y_pred_male = y_pred[male_indices]
y_test_male = y_test[male_indices]

y_pred_female = y_pred[female_indices]
y_test_female = y_test[female_indices]

# Evaluate males
print_metrics(y_test_male, y_pred_male, "Male passengers")

# Evaluate females
print_metrics(y_test_female, y_pred_female, "Female passengers")