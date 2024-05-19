import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load your data
data = pd.read_excel('Retail_data_cleaned.xlsx', engine='openpyxl')

# Data preprocessing steps
data['FirstOrderTime'] = pd.to_datetime(data['FirstOrderTime'])
data['hour_of_day'] = data['FirstOrderTime'].dt.hour
data['day_of_week'] = data['FirstOrderTime'].dt.dayofweek
data['month'] = data['FirstOrderTime'].dt.month
data.drop('FirstOrderTime', axis=1, inplace=True)

# Categorical and numeric transformations
categorical_features = ['Country', 'MostExpensiveItemFirstOrder']
numeric_features = ['hour_of_day', 'day_of_week', 'month', 'FirstOrderTotalPrice']
one_hot = OneHotEncoder(sparse_output=False)
scaler = StandardScaler()

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", one_hot, categorical_features),
        ("num", scaler, numeric_features)
    ],
    remainder="passthrough"
)

# Apply preprocessing
X = data.drop('TotalAnnualExpenditure', axis=1)
y = data['TotalAnnualExpenditure']
X_preprocessed = preprocessor.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Add a constant to the model for the intercept
X_train_with_const = sm.add_constant(X_train)

# Check for multicollinearity
vif_data = pd.DataFrame()
vif_data["feature"] = [f"x{i}" for i in range(X_train_with_const.shape[1])]
vif_data["VIF"] = [variance_inflation_factor(X_train_with_const, i) for i in range(X_train_with_const.shape[1])]
print(vif_data)

# Fit the model on train data using a robust covariance type
model = OLS(y_train, X_train_with_const).fit(cov_type='HC3')
print(model.summary())
