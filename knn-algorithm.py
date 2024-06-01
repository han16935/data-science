import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_data(file_path):
    """
    Load Data
    """
    data = pd.read_csv(file_path)
    return data[['BodyweightKg', 'TotalKg']].dropna()

def classify(row):
    """
    Classification Data
    """
    ratio = row['TotalKg'] / row['BodyweightKg']
    if ratio >= 6.75:
        return 'Pro'
    elif ratio >= 4.72:
        return 'Amateur'
    else:
        return 'Beginner'

def visualize_data(data, new_data=None):
    """
    Visualization Data
    """
    # Define New Class
    data['Class'] = data.apply(classify, axis=1)

    # Mapping color
    color_mapping = {'Pro': 'red', 'Amateur': 'blue', 'Beginner': 'green'}
    data['Color'] = data['Class'].map(color_mapping)

    # Scatterplot visualization
    plt.figure(figsize=(10, 6))
    for class_type in ['Pro', 'Amateur', 'Beginner']:
        subset = data[data['Class'] == class_type]
        plt.scatter(subset['BodyweightKg'], subset['TotalKg'], 
                    label=class_type, color=color_mapping[class_type], alpha=0.6)

    # Add new data
    if new_data is not None:
        plt.scatter(new_data['BodyweightKg'], new_data['TotalKg'], color='black', label='New Data', marker='x')

    plt.xlabel('BodyweightKg')
    plt.ylabel('TotalKg')
    plt.title('BodyweightKg vs TotalKg (Classified)')
    plt.legend()

    plt.show()

def knn_classify(data, new_data, k=3):
    """
    K-Nearest Neighbors classification
    """
    # Calculate distances between new_data and all data points
    distances = np.sqrt(np.sum((data[['BodyweightKg', 'TotalKg']] - new_data.values) ** 2, axis=1))
    
    # Get the indices of the k nearest neighbors
    nearest_indices = distances.argsort()[:k]
    
    # Get the classes of the nearest neighbors
    nearest_classes = data.iloc[nearest_indices]['Class']
    
    # Count the occurrences of each class
    class_counts = nearest_classes.value_counts()
    
    # Return the class with the highest count
    return class_counts.idxmax()

# Load data
data = load_data("openpowerlifting.csv")

# Add new data
new_data = pd.DataFrame({'BodyweightKg': [65], 'TotalKg': [210]})
visualize_data(data, new_data=new_data)

# Classification new data using KNN algorithm
knn_class = knn_classify(data, new_data)
print("New Data is classified as:", knn_class)
