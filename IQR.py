import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data[['Sex', 'BodyweightKg', 'TotalKg']].dropna()

def find_outliers_iqr(data, group_by_column, target_column):
    outliers = pd.DataFrame()
    
    # Group the data by the specified column
    grouped = data.groupby(group_by_column)
    
    # Iterate over each group
    for name, group in grouped:
        Q1 = group[target_column].quantile(0.25)
        Q3 = group[target_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        group_outliers = group[(group[target_column] < lower_bound) | (group[target_column] > upper_bound)]
        outliers = pd.concat([outliers, group_outliers])
    
    return outliers

def plot_outliers_with_new_data(data, group_by_column, target_column, outliers, new_data):
    plt.figure(figsize=(14, 7))
    
    # Plot all data points
    sns.boxplot(x=group_by_column, y=target_column, data=data, showfliers=False)
    sns.stripplot(x=group_by_column, y=target_column, data=data, color='blue', alpha=0.5, jitter=True)
    
    # Highlight outliers
    sns.stripplot(x=group_by_column, y=target_column, data=outliers, color='red', jitter=True, edgecolor='red', linewidth=1)
    
    # Highlight new data
    for _, row in new_data.iterrows():
        plt.scatter(row[group_by_column], row[target_column], color='black', s=100, marker='x', zorder=5)
    
    plt.title(f'Outliers in {target_column} grouped by {group_by_column}')
    plt.xlabel(group_by_column)
    plt.ylabel(target_column)
    plt.show()

# Usage
file_path = './dataset/openpowerlifting.csv'  # Replace with your file path
data = load_data(file_path)
outliers = find_outliers_iqr(data, 'Sex', 'TotalKg')

# Example new data
new_data = pd.DataFrame({
    'Sex': ['M', 'F'],
    'BodyweightKg': [90, 60],
    'TotalKg': [300, 200]
})

plot_outliers_with_new_data(data, 'Sex', 'TotalKg', outliers, new_data)
