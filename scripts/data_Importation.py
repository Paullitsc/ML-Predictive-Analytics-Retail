''' This Script:


'''
import pandas as pd

# Load the data
data = pd.read_excel('Retail_Data.xlsx')  # Update the path to your Excel file

# Calculate total price for each item
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# Correct sorting: earliest to latest
data.sort_values(by='InvoiceDate', inplace=True)

# Calculate total price per order
order_totals = data.groupby('InvoiceNo').agg({
    'TotalPrice': 'sum', 
    'CustomerID': 'first', 
    'Country': 'first', 
    'InvoiceDate': 'first'
}).reset_index()

# Find the first order for each customer
first_order = order_totals.drop_duplicates(subset=['CustomerID'], keep='first')

# Calculate total annual expenditure per customer
annual_expenditure = data.groupby('CustomerID')['TotalPrice'].sum().reset_index()
annual_expenditure.rename(columns={'TotalPrice': 'TotalAnnualExpenditure'}, inplace=True)

# Merge first order data with annual expenditure
final_data = pd.merge(first_order, annual_expenditure, on='CustomerID')

# Find the most expensive item for each first order
first_order_expensive_items = data.loc[data.groupby('InvoiceNo')['TotalPrice'].idxmax()]
first_order_expensive_items = first_order_expensive_items[['InvoiceNo', 'Description']]

# Merge the most expensive item information
final_data = pd.merge(final_data, first_order_expensive_items, on='InvoiceNo', how='left')
final_data.rename(columns={'Description': 'MostExpensiveItemFirstOrder'}, inplace=True)

# Keep and rename necessary columns
final_data = final_data[['CustomerID', 'Country', 'InvoiceDate', 'TotalPrice', 'TotalAnnualExpenditure', 'MostExpensiveItemFirstOrder']]
final_data.columns = ['CustomerID', 'Country', 'FirstOrderTime', 'FirstOrderTotalPrice', 'TotalAnnualExpenditure', 'MostExpensiveItemFirstOrder']

# Save to a new Excel file
final_data.to_excel('updated_output.xlsx', index=False)
