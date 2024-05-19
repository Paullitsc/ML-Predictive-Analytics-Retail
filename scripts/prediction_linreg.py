import pandas as pd
from sklearn.preprocessing import StandardScaler
from model_training import preprocessor, scaler_y, ridge_cv

def preprocess_data(df):
    # Convert 'FirstOrderTime' into datetime and extract features
    df['FirstOrderTime'] = pd.to_datetime(df['FirstOrderTime'])
    df['hour_of_day'] = df['FirstOrderTime'].dt.hour
    df['day_of_week'] = df['FirstOrderTime'].dt.dayofweek
    df['month'] = df['FirstOrderTime'].dt.month
    df.drop('FirstOrderTime', axis=1, inplace=True)

    # Transform the input data using the same preprocessor
    transformed_input = preprocessor.transform(df)

    # Ensure the transformed input is dense
    transformed_input_dense = transformed_input if not hasattr(transformed_input, "toarray") else transformed_input.toarray()

    # Print shapes for debugging
    print(f"Shape of transformed input: {transformed_input_dense.shape}")

    # Convert to DataFrame to maintain feature names
    columns_transformed = preprocessor.named_transformers_['cat'].get_feature_names_out()
    numeric_transformed = df.drop(['Country', 'MostExpensiveItemFirstOrder'], axis=1).columns.tolist()
    all_columns = list(columns_transformed) + numeric_transformed

    # Print column lengths for debugging
    print(f"Number of categorical columns: {len(columns_transformed)}")
    print(f"Number of numeric columns: {len(numeric_transformed)}")
    print(f"Total columns expected: {len(all_columns)}")

    transformed_df = pd.DataFrame(transformed_input_dense, columns=all_columns)

    return transformed_df

def make_predictions(input_file, output_file):
    # Read input data from Excel file
    data = pd.read_excel(input_file, engine='openpyxl')

    # Print the DataFrame to ensure it has been read correctly
    print("Initial data read from file:")
    print(data)

    # Print columns for debugging
    print("Columns in the input data:", data.columns)
    print("Data preview:")
    print(data.head())  # Print first few rows for debugging

    # Check if the DataFrame is empty
    if data.empty:
        print("No data available for predictions.")
        return

    # Preprocess the data
    X_transformed = preprocess_data(data)

    # Standardize the input data using the same scaler from training
    X_scaled = StandardScaler().fit_transform(X_transformed)

    # Convert back to DataFrame to retain column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_transformed.columns)

    # Make predictions using the best model (RidgeCV)
    predictions_scaled = ridge_cv.predict(X_scaled_df)

    # Inverse transform the scaled predictions to get the original scale
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Post-processing predictions
    half_year_expenditure = data['HalfYearExpenditure']
    predictions = [
        hy_exp if pred < 0 else max(pred, 1.85 * hy_exp)
        for pred, hy_exp in zip(predictions, half_year_expenditure)
    ]

    # Add predictions to the original DataFrame
    data['PredictedAnnualExpenditure'] = predictions

    # Write the DataFrame with predictions back to a new Excel file
    data.to_excel(output_file, index=False)

    print(f"Predictions have been written to {output_file}")

# Example usage
input_file = 'input_data.xlsx'
output_file = 'output_data_with_predictions.xlsx'
make_predictions(input_file, output_file)
