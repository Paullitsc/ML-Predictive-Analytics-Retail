# =============================================================================
# Imports
# =============================================================================
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
def load_and_preprocess_data(file_path='final.xlsx'):
    data = pd.read_excel(file_path, engine='openpyxl')

    # Convert 'FirstOrderTime' into datetime and extract features
    data['FirstOrderTime'] = pd.to_datetime(data['FirstOrderTime'])
    data['hour_of_day'] = data['FirstOrderTime'].dt.hour
    data['day_of_week'] = data['FirstOrderTime'].dt.dayofweek
    data['month'] = data['FirstOrderTime'].dt.month
    data.drop('FirstOrderTime', axis=1, inplace=True)

    # Define transformations for categorical and numerical data
    categorical_features = ['Country', 'MostExpensiveItemFirstOrder']
    numeric_features = [col for col in data.columns if col not in categorical_features + ['TotalAnnualExpenditure']]

    # Setup the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )

    # Create a pipeline with the preprocessor
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Apply transformations
    transformed_X = pipeline.fit_transform(data.drop('TotalAnnualExpenditure', axis=1))

    # Convert back to DataFrame
    columns_transformed = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
    numeric_transformed = data[numeric_features].columns.tolist()
    all_columns = list(columns_transformed) + numeric_transformed

    transformed_df = pd.DataFrame(transformed_X.toarray(), columns=all_columns)

    X = transformed_df
    y = data['TotalAnnualExpenditure']

    # Standardize the target variable
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    return X, y_scaled, preprocessor, scaler_y

X, y_scaled, preprocessor, scaler_y = load_and_preprocess_data()

# Split the data into 'train' and 'test' sets
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# =============================================================================
# Model Training and Evaluation - Ridge Regression
# =============================================================================
alphas = np.logspace(-6, 2, 100)  # Adjusting alpha range for Ridge and Lasso
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv.fit(X_train, y_train)

y_pred_ridge = ridge_cv.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Cross-Validation for Ridge
ridge_scores = cross_val_score(ridge_cv, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# =============================================================================
# Model Training and Evaluation - Lasso Regression
# =============================================================================
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=20000)
lasso_cv.fit(X_train, y_train)

y_pred_lasso = lasso_cv.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# =============================================================================
# Model Training and Evaluation - ElasticNet Regression
# =============================================================================
elastic_net_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], alphas=alphas, cv=5, max_iter=20000)
elastic_net_cv.fit(X_train, y_train)

y_pred_elastic = elastic_net_cv.predict(X_test)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
rmse_elastic = np.sqrt(mse_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)

# =============================================================================
# Diagnostic Plots and Evaluation
# =============================================================================
'''
## Evaluation for Ridge Regression
print(f'Best alpha for Ridge: {ridge_cv.alpha_}')
print(f'Ridge MSE: {mse_ridge}')
print(f'Ridge RMSE: {rmse_ridge}')
print(f'Ridge R-squared: {r2_ridge}')
# Ridge Cross-Validation
print(f'Ridge Cross-Validation MSE: {-ridge_scores.mean()}')


## Residual Plot for Ridge Regression
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_ridge, y_test - y_pred_ridge)
plt.title('Residual Plot for Ridge Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

## QQ Plot for Ridge Regression
plt.figure(figsize=(10, 5))
stats.probplot(y_test - y_pred_ridge, dist="norm", plot=plt)
plt.title('QQ Plot for Ridge Regression')
plt.show()


## Evaluation for Lasso Regression
print(f'Best alpha for Lasso: {lasso_cv.alpha_}')
print(f'Lasso MSE: {mse_lasso}')
print(f'Lasso RMSE: {rmse_lasso}')
print(f'Lasso R-squared: {r2_lasso}')

## Residual Plot for Lasso Regression
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_lasso, y_test - y_pred_lasso)
plt.title('Residual Plot for Lasso Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

## QQ Plot for Lasso Regression
plt.figure(figsize=(10, 5))
stats.probplot(y_test - y_pred_lasso, dist="norm", plot=plt)
plt.title('QQ Plot for Lasso Regression')
plt.show()


## Evaluation for Elastic Net Regression
print(f'Best alpha for Elastic Net: {elastic_net_cv.alpha_}')
print(f'Best l1_ratio for Elastic Net: {elastic_net_cv.l1_ratio_}')
print(f'Elastic Net MSE: {mse_elastic}')
print(f'Elastic Net RMSE: {rmse_elastic}')
print(f'Elastic Net R-squared: {r2_elastic}')

## Residual Plot for Elastic Net Regression
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_elastic, y_test - y_pred_elastic)
plt.title('Residual Plot for Elastic Net Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

## QQ Plot for Elastic Net Regression
plt.figure(figsize=(10, 5))
stats.probplot(y_test - y_pred_elastic, dist="norm", plot=plt)
plt.title('QQ Plot for Elastic Net Regression')
plt.show()
'''
