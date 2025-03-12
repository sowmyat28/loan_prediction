# loan Default Analysis Using machine learning 
## Data Set Problems 
The company seeks to automate (in real time) the loan qualifying procedure based on information given by customers while filling out an online application form. It is expected that the development of ML models that can help the company predict loan approval in accelerating decision-making process for determining whether an applicant is eligible for a loan or not.
##  1. Introduction

### Objective: 
Automate the loan approval process using machine learning models to predict loan eligibility in real-time.
### Models Used:
Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Naive Bayes, Decision Tree, Random Forest, and Gradient Boosting.
### Dataset:
Contains 13 variables (8 categorical, 4 continuous, and 1 unique ID).

## 2. Data Exploration

### Categorical Variables:
Explored variables like Gender, Married, Education, Self_Employed, Credit_History, Property_Area, and Loan_Status.
Most applicants are male, married, and graduates.
The majority of loans are approved, and most applicants have good credit history.
### Numerical Variables: 
Analyzed ApplicantIncome, CoapplicantIncome, LoanAmount, and Loan_Amount_Term.
Distributions are positively skewed, with outliers present.
### Heatmap: 
Revealed a positive correlation between LoanAmount and ApplicantIncome.

## 3. Data Preprocessing
### Handling Missing Values: 
Imputed missing values using mode for categorical variables and mean for numerical variables.
### Encoding: 
Applied one-hot encoding to categorical variables.
### Outlier Removal:
Removed outliers using the IQR method.
### Skewness Treatment:
Applied square root transformation to normalize skewed distributions.
### SMOTE: 
Balanced the dataset using SMOTE to address class imbalance.
### Normalization:
Scaled features using MinMaxScaler.
### Train-Test Split:
Split the data into 80% training and 20% testing sets.

## 4. Model Building
Logistic Regression: Achieved an accuracy of 73.9644%.
K-Nearest Neighbors (KNN): Best accuracy of 74.55% with k=3.
Support Vector Machine (SVM): Achieved an accuracy of 49.11%.
Naive Bayes:
Categorical NB: 78.10% accuracy.
Gaussian NB: 71.05% accuracy.
Decision Tree: Best accuracy of 81.65% with max_leaf_nodes=10.
Random Forest: Best accuracy of 84.61% with max_leaf_nodes=20.
Gradient Boosting: Achieved the highest accuracy of 82.84% with optimized hyperparameters.

## Key Findings
The Random Forest model performed the best with an accuracy of 84.61%.
Gradient Boosting and Decision Tree also performed well, achieving accuracies above 70%.
The dataset had imbalanced classes, which was addressed using the SMOTE technique.
Outliers and skewed distributions were handled effectively during preprocessing.

## Suggestions for Improvement
### Hyperparameter Tuning:
Perform more extensive hyperparameter tuning for models like Random Forest and Gradient Boosting to potentially improve accuracy further.
### Feature Engineering:
Create new features (e.g., total income = ApplicantIncome + CoapplicantIncome) to capture more information.
### Advanced Models:
Experiment with advanced models like XGBoost, LightGBM, or Neural Networks.
### Cross-Validation:
Use k-fold cross-validation to ensure the robustness of the models.
### Explainability:
Use SHAP or LIME to explain model predictions, especially for stakeholders.

## Next Steps
Deploy the best-performing model (Random Forest) as a real-time loan approval system.
Monitor the model's performance in production and retrain periodically with new data.
