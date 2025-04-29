import pandas as pd

# Load the dataset
df = pd.read_csv('travel_cost.csv')

# Display the first few rows of the dataset
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Feature Engineering: Separate the features (X) and target variable (y)
X = df[['location', 'season', 'travelers', 'preference', 'accommodation_type', 'activities']]
y = df['total_cost']

# One-Hot Encoding for categorical columns
categorical_cols = ['location', 'season', 'preference', 'accommodation_type', 'activities']
numerical_cols = ['travelers']

# Column transformer to handle categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
