'''
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned, processed data from Excel
df = pd.read_excel("Retail_data_cleaned.xlsx")

# Create frequency distribution data with 20 bins from 0 to 5000
bin_edges = np.linspace(0, 5000, 21)
bin_labels = (bin_edges[:-1] + bin_edges[1:]) / 2
freq, _ = np.histogram(df["TotalAnnualExpenditure"], bins=bin_edges)

# Create a line plot of the frequency distribution
plt.figure(figsize=(8, 6))
plt.plot(bin_labels, freq, marker='o', linestyle='-', color="skyblue")
plt.title("Line Plot of Frequency Distribution for Annual Expenditure")
plt.xlabel("Annual Expenditure")
plt.ylabel("Frequency")
plt.axvline(df["TotalAnnualExpenditure"].mean(), color='purple', linestyle='--', label='Mean')
plt.axvline(df["TotalAnnualExpenditure"].median(), color='darkblue', linestyle='--', label='Median')
plt.legend()
plt.show()

# Time graph of first order impacts on Annual Expenditure
plt.figure(figsize=(10, 6))
# Filter out outliers beyond 3 standard deviations from the mean
upper_limit = df["TotalAnnualExpenditure"].mean() + 3 * df["TotalAnnualExpenditure"].std()
df_filtered = df[df["TotalAnnualExpenditure"] <= upper_limit]

sns.lineplot(x='FirstOrderTime', y='TotalAnnualExpenditure', data=df_filtered)
plt.title("Effect of the Time Of First Orders on Annual Expenditure")
plt.xlabel("Time")
plt.ylabel("Annual Expenditure")
plt.xlim(df_filtered['FirstOrderTime'].min(), df_filtered['FirstOrderTime'].max())
plt.ylim(0, df_filtered['TotalAnnualExpenditure'].max())
plt.show()

# Average and median annual expenditure for top 10 most popular first order items
top_10_items = df["MostExpensiveItemFirstOrder"].value_counts().head(10).index
average_expenditure = []
median_expenditure = []
for item in top_10_items:
    item_data = df[df["MostExpensiveItemFirstOrder"] == item]["TotalAnnualExpenditure"]
    average_expenditure.append(item_data.mean())
    median_expenditure.append(item_data.median())

plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_items, y=average_expenditure, palette="viridis", alpha=0.7)
plt.title("Average Annual Expenditure for Top 10 Most Popular Items")
plt.xlabel("Item")
plt.ylabel("Average Annual Expenditure")
for i, item in enumerate(top_10_items):
    plt.plot([i - 0.2, i + 0.2], [median_expenditure[i], median_expenditure[i]], color='red', linestyle='-', linewidth=2,
             label='Median' if i == 0 else "")
plt.xticks(rotation=90)
plt.legend()
plt.show()