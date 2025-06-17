Project Report
Title: Survival Prediction of Cancer Patients Using Machine Learning 
________________________________________
1. Introduction
In recent years, predictive modeling in healthcare has become increasingly important. Predicting patient outcomes such as survival can provide doctors with valuable insights and help improve treatment planning. In this project, we focus on developing a machine learning pipeline that predicts whether a cancer patient is likely to survive based on various medical and demographic features.
The main objectives of the project were: - To perform thorough exploratory data analysis and feature engineering. - To preprocess and clean the dataset effectively. - To apply multiple machine learning models and tune them. - To compare model performances and select the best-performing one. - To achieve an F1 score and accuracy greater than 80%.
________________________________________
2. Dataset Overview
We worked with two primary datasets: - Train Dataset: Contains data on cancer patients including demographics, medical history, lifestyle, and treatment details. Target variable is Survival Prediction. - Test Dataset: Used for final prediction and Kaggle submission.
The target variable is binary: - Yes (1): Patient survived - No (0): Patient did not survive
The dataset contains a rich set of features: - Demographics (Age, Gender, Country, Urban/Rural, etc.) - Lifestyle (Smoking, Alcohol, Physical Activity, Diet Risk, etc.) - Medical (Tumor size, Cancer stage, Diagnosis delay, Insurance) - Treatment (Type of treatment, Transfusion, Screening history, etc.)
________________________________________
3. Data Cleaning & Preprocessing
Raw data needed significant preprocessing: - Categorical to Numerical Conversion: - Yes/No → 1/0 - Gender: M → 1, F → 0 - Urban or Rural: Urban → 1, Rural → 0 - Obesity BMI, Cancer Stage, Diet Risk: Encoded numerically - Date Handling: - Date of Birth was converted to Age using the difference from today’s date. - Missing Values: - Filled Alcohol Consumption with 0 - Screening History → NaN filled with “Never” - Dropped rows with excessive missing values - Dropped the column Marital Status which wasn’t useful
________________________________________
4. Feature Engineering
New features were created to improve model learning:
•	Age: Derived from Date of Birth
•	Geo-Area: Based on the patient’s country, classified as:
o	Low-Resource (e.g., India, Nigeria)
o	Medium-Resource (e.g., Brazil, China)
o	High-Resource (e.g., USA, UK)
•	Tumor Size Range: Binned into 5 categories:
o	(0–31 mm), (31–62 mm), (62–93 mm), (93–124 mm), (>124 mm)
These features helped to expose important nonlinear patterns in the data.
________________________________________
5. Exploratory Data Analysis (EDA)
Several univariate and bivariate analyses were done: - Distribution of survivors vs. non-survivors - Impact of treatment type and cancer stage on survival - Correlation analysis using Kendall method - Observed trends: - Patients with Metastatic stage had low survival - Combination treatments had higher survival rates - Age and tumor size were negatively correlated with survival
________________________________________
6. Feature Selection
To reduce noise and improve generalization: - Mutual Information was used to measure feature importance - Only features with score > 0.01 were selected for training
________________________________________
7. Class Imbalance Handling
Original class distribution was imbalanced (more survivors): - Applied SMOTE (Synthetic Minority Oversampling Technique) to balance classes - Ensured training data had equal representation of both classes
________________________________________
8. Model Building
Initially several models were tested: - Logistic Regression - Decision Tree - Random Forest - XGBoost
After experimentation: - XGBoostClassifier gave the best results - Robust to multicollinearity and works well with tabular data
________________________________________
9. Hyperparameter Tuning
We used RandomizedSearchCV with 5-fold cross-validation to tune: - n_estimators: Number of trees - max_depth: Depth of each tree - learning_rate: Step size shrinkage - subsample, colsample_bytree: Sampling ratios
Tuning was done on resampled data with SMOTE to avoid bias.
________________________________________
10. Final Model Evaluation
Performance on validation set: - F1 Score: 82.6% - Accuracy: 82.2% - Precision: 84.4% - Recall: 82.2%
This met our target and indicated a well-balanced model.
Confusion matrix and classification report were also generated to inspect false positives and false negatives.
________________________________________
11. Final Predictions and Kaggle Submission
•	Final model was used to predict on the patient_test_data.csv
•	Predictions formatted as:
o	Patient_ID, Survival Prediction
•	Saved in CSV and submitted to Kaggle
________________________________________
12. Key Learnings
•	Importance of preprocessing and feature engineering
•	Class imbalance must be addressed before training
•	Tuning hyperparameters drastically improves results
•	XGBoost is very effective for structured classification tasks
________________________________________
13. Limitations
•	Dataset size is moderate
•	Some features could be noisy or biased
•	Limited access to clinical history or longitudinal data
________________________________________
14. Future Work
•	Try ensemble models combining XGBoost, CatBoost, and RandomForest
•	Use SHAP values for interpretability
•	Apply deep learning on a richer dataset
•	Explore survival regression instead of classification
________________________________________
15. Conclusion
This project demonstrates how to use data-driven approaches for survival prediction in cancer patients. Using XGBoost, with careful preprocessing and feature selection, we were able to build a model with over 80% accuracy and F1 score.
The pipeline is reproducible and deployable with potential real-world use in clinical decision support systems.
________________________________________
Appendix
•	Python version: 3.10+
•	Libraries: pandas, numpy, sklearn, xgboost, imblearn
•	SMOTE: from imblearn.over_sampling
•	Modeling: XGBClassifier from xgboost
•	Tuning: RandomizedSearchCV
•	Feature Selection: mutual_info_classif
________________________________________

