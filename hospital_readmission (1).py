
#DATA UNDERSTANDING

# SECTION A: DATA UNDERSTANDING

# Import necessary libraries
# pandas: For creating DataFrames, handling data
# numpy: For numerical operations and array handling
import pandas as pd
import numpy as np

# Load Dataset manually
# Creating a dictionary containing patient data
#'None' represents missing/empty values in the data

data = {'Patientid': ['P001' , 'P002' , 'P003' , 'P004' , 'P005' , 'P006' , 'P007' , 'P008' , 'P009' , 'P010' , 'P011' , 'P012' , 'P013', 'P014' , 'P015' , 'P016' , 'P017' , 'P018' , 'P019' , 'P020' ],
        'Age': [25,67,54,38,72, 49, 59, 46, 81, 34, None, 58, 64, 29, 70, 42, 56, 48, 75, 36],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'BMI': [24.5, 31.2, None, 27.0, 29.8, 26.4, 33.1, 28.0, 30.5, 23.8, 29.0, 27.5, 32.2, 24.1, 34.0, 26.8, None, 27.2, 31.5, 25.0],
        'BloodPressure': [120, 145, 138, 125, 150, 130, None, 135, 155, 118, 140, 132, 148, 122, 152, 128, 140, 134, 150, 124],
        'Cholesterol': [180, 240, 220, None, 260, 210, 245, 215, 270, 175, 230, 218, None, 182, 255, 205, 225, 210, 265, 190],
        'Diabetes': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'SmokingStatus': ['Non-Smoker', 'Former', 'Smoker', 'Non-Smoker', 'Former', 'Non-Smoker', 'Smoker', 'Non-Smoker', 'Former', 'Non-Smoker',
                          'Smoker', 'Former', 'Smoker', 'Non-Smoker', 'Former', 'Non-Smoker', 'Smoker', 'Former', 'Former', 'Non-Smoker'],
        'NumVisitsLastYear': [1, 5, 4, 2, 6, 3, 5, None, 7, 1, 4, 3, 6, 1, 6, 2, 5, 3, 7, 2],
        'HospitalStayDays': [2, 10, 8, 3, 12, 4, 9, 5, 15, 2, 7, 5, 11, 2, 13, 3, 9, 4, 14, 3],
        'MedicationCount': [2, 6, 5, 3, 7, 4, 6, 4, 8, 2, 5, 4, 7, 2, 7, 3, 6, 4, 8, 3],
        'InsuranceType': ['Public', 'Private', 'Public', 'Public', 'Private', 'Public', 'Private','Public','Private', 'Public', 'Public', 'Public', 'Private', 'Public', 'Private', 'Public','Private', 'Public',
                           'Private', 'Public'],
        'City': ['Dhaka', 'Chattogram', 'Dhaka', 'Sylhet', 'Rajshahi', 'Dhaka', 'Chattogram', 'Khulna','Sylhet','Dhaka', 'Rajshahi', 'Dhaka', 'Khulna', 'Dhaka', 'Chattogram', 'Dhaka', 'Rajshahi',
                 'Sylhet', 'Khulna', 'Dhaka'],
                'Readmitted': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No','Yes', 'No', 'Yes', 'No', 'Yes', 'No']}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Analyzed dataset shape (20 patients × 14 features)
# Checked data types and missing values
print("\n1. Dataset Shape:")
print(f"   - Rows (patients): {df.shape[0]}")
print(f"   - Columns (features): {df.shape[1]}")
print("\n2. Dataset Info :")
print(df.info())
print("\n4. Missing Values :")
print(df.isnull().sum())
print()
#Identified which columns are numeric vs categorical
numeric_cols = []
categorical_cols = []
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)
print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)


#DATA CLEANING

# SECTION B: DATA CLEANING

# Created a working copy (preserve original data)
df_clean = df.copy()

# Dropped irrelevant column (Patientid - not useful for prediction)
df_clean = df_clean.drop('Patientid', axis=1)

# Checked target variable for missing values (none found in this dataset)
missing_readmitted = df_clean['Readmitted'].isnull().sum()
if missing_readmitted > 0:
    df_clean = df_clean.dropna(subset=['Readmitted'])

# Imputed numeric columns with MEAN values
# Imputed categorical columns with MODE (most frequent value)
numeric_cols_for_impute = ['Age', 'BMI', 'BloodPressure', 'Cholesterol',
                           'NumVisitsLastYear', 'HospitalStayDays', 'MedicationCount']
for col in numeric_cols_for_impute:
    if col in df_clean.columns:
        mean_value = df_clean[col].mean()
        df_clean[col] = df_clean[col].fillna(mean_value)
categorical_cols_for_impute = ['Gender', 'Diabetes', 'SmokingStatus', 'InsuranceType', 'City']
for col in categorical_cols_for_impute:
    if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
        mode_value = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_value)
print("Dataset after cleaning:")
print(f"Shape: {df_clean.shape}")
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())
# RESULT: Clean dataset ready for encoding and modeling

#ENCODING & FEATURE ENGINEERING


# Import OneHotEncoder for categorical variable encoding
from sklearn.preprocessing import OneHotEncoder

# Define categorical columns for encoding
categorical_cols = ['Gender', 'Diabetes', 'SmokingStatus', 'InsuranceType', 'City']

# Create OneHotEncoder instance
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Encode categorical columns to binary arrays
encoded_array = encoder.fit_transform(df_clean[categorical_cols])

# Get encoded column names
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

# Create DataFrame for encoded features
encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=df_clean.index)

# Remove original categorical columns
df_encoded = df_clean.drop(categorical_cols, axis=1)

# Add encoded columns to DataFrame
df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

# Convert target variable to binary
df_encoded['Readmitted'] = df_encoded['Readmitted'].map({'Yes': 1, 'No': 0})

# Create RiskScore feature
df_encoded['RiskScore'] = (df_encoded['BMI'] + df_encoded['BloodPressure'] + df_encoded['Cholesterol']) / 3

# Print dataset information
print("Dataset after encoding and feature engineering:")
print(f"Shape: {df_encoded.shape}")
print(f"\nEncoded columns created: {len(encoded_feature_names)}")
print(f"\nTarget variable values after encoding: {df_encoded['Readmitted'].unique()}")
print(f"\nRiskScore created. Sample values:")
print(df_encoded[['BMI', 'BloodPressure', 'Cholesterol', 'RiskScore']].head())

#SCALING & SPLITTING

# SECTION D: SCALING & SPLITTING
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Separate features and target variable
X = df_encoded.drop('Readmitted', axis=1)
y = df_encoded['Readmitted']

# Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=2
)

# Print results
print("Scaling and splitting completed:")
print(f"Original features shape: {X.shape}")
print(f"Scaled features shape: {X_scaled.shape}")
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Random state: 2")

#MODEL TRAINING

# SECTION E: MODEL TRAINING
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Initialize and train Logistic Regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# Initialize and train KNN model with k=7
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)

# Print training status
print("Model training completed")
print("Logistic Regression: trained successfully")
print("KNN (k=7): trained successfully")

#EVALUATION METRICS

# SECTION F: EVALUATION METRICS
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Make predictions on test set
y_pred_logistic = logistic_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

# Calculate confusion matrices
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
cm_knn = confusion_matrix(y_test, y_pred_knn)

print("Confusion Matrices:")
print(f"Logistic Regression:\n{cm_logistic}")
print(f"\nKNN (k=7):\n{cm_knn}")

# Calculate accuracy scores
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"\nAccuracy:")
print(f"Logistic Regression: {accuracy_logistic:.4f}")
print(f"KNN (k=7): {accuracy_knn:.4f}")

# Calculate precision, recall, and F1 scores
precision_logistic = precision_score(y_test, y_pred_logistic)
recall_logistic = recall_score(y_test, y_pred_logistic)
f1_logistic = f1_score(y_test, y_pred_logistic)

precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

print(f"\nPrecision, Recall, F1-Score:")
print(f"Logistic Regression - Precision: {precision_logistic:.4f}, Recall: {recall_logistic:.4f}, F1: {f1_logistic:.4f}")
print(f"KNN (k=7) - Precision: {precision_knn:.4f}, Recall: {recall_knn:.4f}, F1: {f1_knn:.4f}")

# Calculate ROC curve and AUC
y_pred_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]
y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]

fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_proba_logistic)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_proba_knn)

auc_logistic = auc(fpr_logistic, tpr_logistic)
auc_knn = auc(fpr_knn, tpr_knn)

print(f"\nROC AUC:")
print(f"Logistic Regression: {auc_logistic:.4f}")
print(f"KNN (k=7): {auc_knn:.4f}")

#VISUALIZATION

# SECTION G: VISUALIZATION
import matplotlib.pyplot as plt

# Plot ROC curves for both models
plt.figure(figsize=(8, 6))
plt.plot(fpr_logistic, tpr_logistic, label=f'Logistic Regression (AUC = {auc_logistic:.3f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (k=7) (AUC = {auc_knn:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()

# Prepare data for bar chart comparison
models = ['Logistic Regression', 'KNN (k=7)']
accuracy_values = [accuracy_logistic, accuracy_knn]
auc_values = [auc_logistic, auc_knn]

# Plot bar chart comparing accuracy and AUC
plt.figure(figsize=(8, 6))
x = range(len(models))
width = 0.35
plt.bar([i - width/2 for i in x], accuracy_values, width, label='Accuracy')
plt.bar([i + width/2 for i in x], auc_values, width, label='AUC')
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, models)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#ANALYTICAL REPORT

# SECTION H: ANALYTICAL REPORT

print("ANALYTICAL REPORT")

print("\n1. Factors most influencing readmission:")
# Logistic Regression coefficients for feature importance
coefficients = logistic_model.coef_[0]
feature_names = X.columns

# Get top 5 most influential features
feature_importance = list(zip(feature_names, abs(coefficients)))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("Top 5 influential features (based on Logistic Regression coefficients):")
for i, (feature, importance) in enumerate(feature_importance[:5], 1):
    print(f"   {i}. {feature}")

print("\n2. More reliable model:")
print(f"   Logistic Regression Accuracy: {accuracy_logistic:.4f}")
print(f"   KNN (k=7) Accuracy: {accuracy_knn:.4f}")
print(f"   Logistic Regression AUC: {auc_logistic:.4f}")
print(f"   KNN (k=7) AUC: {auc_knn:.4f}")

if accuracy_logistic > accuracy_knn and auc_logistic > auc_knn:
    print("   → Logistic Regression is more reliable")
elif accuracy_knn > accuracy_logistic and auc_knn > auc_logistic:
    print("   → KNN (k=7) is more reliable")
else:
    print("   → Models show mixed performance")