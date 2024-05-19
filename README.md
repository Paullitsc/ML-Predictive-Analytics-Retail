# Machine Learning Predictive Analytics in Retail

## Project Description
Developed a machine learning pipeline in Python to predict annual customer expenditure using data inputs such as country and price of first order. The project integrates advanced data preprocessing, exploratory data analysis, and feature engineering.

## Key Features
- **Data Preprocessing:** Cleaned and prepared raw data for analysis.
- **Exploratory Data Analysis:** Performed detailed exploratory analysis on the dataset.
- **Feature Engineering:** Engineered features to improve model performance.
- **Model Training:** Implemented robust regression models for precise predictive analytics.
- **Prediction Output:** Generated predictions and outputted results to an Excel file for seamless business integration.

## Files
- `scripts/data_Importation.py`: Imports and preprocesses data.
- `scripts/exploratory_analysis.py`: Performs exploratory analysis on the raw data.
- `scripts/improve_data.py`: Improves the data quality.
- `scripts/model_training.py`: Trains the linear regression model.
- `scripts/prediction_linreg.py`: Utilizes the trained model on an input Excel sheet and outputs predictions to an output Excel sheet.
- `scripts/removal.py`: Removes unnecessary data that affects the linear regression model.

## Data
- `data/input_data.xlsx`: Input data for predictions.
- `data/final.xlsx`: Final processed data.
- `data/output_data_with_predictions.xlsx`: Output data with predictions.
- `data/Possible_Cat_Inputs.xlsx`: Possible categorical inputs.
- `data/Original_data.xlsx`: Uncleaned Retail data.
## User Interaction Example
Input Sheet
![Screenshot 2024-05-19 140925](https://github.com/Paullitsc/ML-Predictive-Analytics-Retail/assets/168594999/6adffc7f-7765-4a2d-bd8e-4f122b541cfe)
Output Sheet
![Screenshot 2024-05-19 141014](https://github.com/Paullitsc/ML-Predictive-Analytics-Retail/assets/168594999/872507e1-c76c-467f-801a-fea1e019b319)

## Development
### Developed Using
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl

### Compiliation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ML-Predictive-Analytics-Retail.git
   cd ML-Predictive-Analytics-Retail
   ```

2. Install the required libraries:
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
  ```

3. Open `data/input_data.xlsx` and enter the data you want to have a prediction for. If you need legible country names or the most expensive first products, refer to `data/Possible_Cat_Inputs.xlsx`. Make sure to save the file.

4. Run the linear Regression Ridge model on the data input:
  ```bash
  python scripts/prediction_linreg.py
  ```

5. You can see the predicted annual expenditure of the given inputs in `data/output_data_with_predictions.xlsx`
