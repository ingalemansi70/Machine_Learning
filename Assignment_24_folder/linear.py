import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
linear_df = pd.read_csv("../aiml_salary.csv")
binary_df = pd.read_csv("Heart_Dataset.csv")
print("Dataset for linear model : ")
print(linear_df.head(2))
print("\nDataset for classification model : ")
print(binary_df.head(2))

linear_df=linear_df.drop(["job_id","job_title","industry","experience_level","city","posting_year","demand_growth_yoy_pct","posting_month","is_remote_friendly","is_llm_role","ai_salary_premium_pct","remote_work","company_size","benefits_score_10","salary_tier","country","demand_score"], axis=1)
print("Cleaned dataset size : ",linear_df.shape)
linear_df.head()


linear_y= linear_df["annual_salary_usd"]
linear_X = linear_df.drop(["annual_salary_usd","salary_min_usd","salary_max_usd"],axis=1)
print("Target columns are : ",linear_y.columns)
print("Input columns are : ",linear_X.columns)
print("Columns in X right now:")
print(linear_X.columns.tolist())

linear_cat_cols = ["job_category", "education_required", "required_skills"]  
linear_X = pd.get_dummies(data = linear_X, columns=cat_cols, dtype=int)

print("Dataset after one hot encoding : ")
print(linear_X.head())
from sklearn.preprocessing import StandardScaler
linear_num_cols = ["years_of_experience"]

linear_scaler = StandardScaler()
linear_X[num_cols] = linear_scaler.fit_transform(linear_X[num_cols])

print("After scaling:")
print(linear_X[linear_num_cols].describe())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(linear_X, linear_y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
