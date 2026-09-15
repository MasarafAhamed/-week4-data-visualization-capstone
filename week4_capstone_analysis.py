"""
Skill Nexis Week 4 - End-to-End Capstone
EDA + Cleaning + Regression Model

Input:
    Sample data (1).xlsx

Outputs:
    week4_cleaned_data.csv
    week4_regression_predictions.csv

The script:
1. Loads and cleans the sales dataset.
2. Handles missing Discount Band values.
3. Removes exact duplicates.
4. Creates Profit Margin % and Discount %.
5. Performs grouped EDA.
6. Trains a linear regression model to predict Sales.
7. Reports MAE, RMSE and R².
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel("Sample data (1).xlsx")
df.columns = [c.strip() for c in df.columns]

for c in ["Segment", "Country", "Product", "Discount Band", "Month Name"]:
    df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

df["Discount Band"] = df["Discount Band"].fillna("Not Specified")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_cols = [
    "Units Sold", "Manufacturing Price", "Sale Price", "Gross Sales",
    "Discounts", "Sales", "COGS", "Profit", "Month Number", "Year"
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.drop_duplicates().copy()
df["Profit Margin %"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"] * 100, np.nan)
df["Discount %"] = np.where(df["Gross Sales"] != 0, df["Discounts"] / df["Gross Sales"] * 100, np.nan)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\\nMissing values:")
print(df.isna().sum())

print("\\nSales by Segment:")
print(df.groupby("Segment")["Sales"].sum().sort_values(ascending=False))

features = [
    "Units Sold", "Sale Price", "Discounts", "Manufacturing Price",
    "Month Number", "Year", "Segment", "Country", "Product", "Discount Band"
]
X = df[features]
y = df["Sales"]

numeric_features = [
    "Units Sold", "Sale Price", "Discounts", "Manufacturing Price",
    "Month Number", "Year"
]
categorical_features = ["Segment", "Country", "Product", "Discount Band"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\\nRegression Results")
print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R²  :", r2_score(y_test, pred))

predictions = X_test.copy()
predictions["Actual Sales"] = y_test.values
predictions["Predicted Sales"] = pred
predictions["Absolute Error"] = np.abs(
    predictions["Actual Sales"] - predictions["Predicted Sales"]
)

df.to_csv("week4_cleaned_data.csv", index=False)
predictions.to_csv("week4_regression_predictions.csv", index=False)
print("\\nSaved cleaned data and regression predictions.")
