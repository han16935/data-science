import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    relevant_columns = ['BestSquatKg', 'BestBenchKg', 'BestDeadliftKg']
    data = data[relevant_columns].dropna()
    return data

def train_models(data):
    models = {
        'squat_to_bench_deadlift': LinearRegression(),
        'bench_to_squat_deadlift': LinearRegression(),
        'deadlift_to_squat_bench': LinearRegression(),
    }
    
    X_squat = data[['BestSquatKg']]
    y_bench_deadlift = data[['BestBenchKg', 'BestDeadliftKg']]
    X_train, X_test, y_train, y_test = train_test_split(X_squat, y_bench_deadlift, test_size=0.2, random_state=42)
    models['squat_to_bench_deadlift'].fit(X_train, y_train)

    X_bench = data[['BestBenchKg']]
    y_squat_deadlift = data[['BestSquatKg', 'BestDeadliftKg']]
    X_train, X_test, y_train, y_test = train_test_split(X_bench, y_squat_deadlift, test_size=0.2, random_state=42)
    models['bench_to_squat_deadlift'].fit(X_train, y_train)

    X_deadlift = data[['BestDeadliftKg']]
    y_squat_bench = data[['BestSquatKg', 'BestBenchKg']]
    X_train, X_test, y_train, y_test = train_test_split(X_deadlift, y_squat_bench, test_size=0.2, random_state=42)
    models['deadlift_to_squat_bench'].fit(X_train, y_train)
    
    return models

def plot_predictions(data, models, sample):
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    # Predictions from Squat
    X_test = np.linspace(data['BestSquatKg'].min(), data['BestSquatKg'].max(), 100).reshape(-1, 1)
    y_pred = models['squat_to_bench_deadlift'].predict(X_test)
    axs[0].plot(X_test, y_pred[:, 0], color='red', label='Predicted Bench')
    axs[0].plot(X_test, y_pred[:, 1], color='orange', label='Predicted Deadlift')
    axs[0].scatter(sample[0], sample[1], color='red', marker='x', s=100)
    axs[0].scatter(sample[0], sample[2], color='orange', marker='x', s=100)
    axs[0].set_xlabel('Squat Weight')
    axs[0].set_ylabel('Bench/Deadlift Weight')
    axs[0].set_title('Predictions from Squat Weight')
    axs[0].legend()

    # Predictions from Bench
    X_test = np.linspace(data['BestBenchKg'].min(), data['BestBenchKg'].max(), 100).reshape(-1, 1)
    y_pred = models['bench_to_squat_deadlift'].predict(X_test)
    axs[1].plot(X_test, y_pred[:, 0], color='red', label='Predicted Squat')
    axs[1].plot(X_test, y_pred[:, 1], color='orange', label='Predicted Deadlift')
    axs[1].scatter(sample[1], sample[0], color='red', marker='x', s=100)
    axs[1].scatter(sample[1], sample[2], color='orange', marker='x', s=100)
    axs[1].set_xlabel('Bench Weight')
    axs[1].set_ylabel('Squat/Deadlift Weight')
    axs[1].set_title('Predictions from Bench Weight')
    axs[1].legend()

    # Predictions from Deadlift
    X_test = np.linspace(data['BestDeadliftKg'].min(), data['BestDeadliftKg'].max(), 100).reshape(-1, 1)
    y_pred = models['deadlift_to_squat_bench'].predict(X_test)
    axs[2].plot(X_test, y_pred[:, 0], color='red', label='Predicted Squat')
    axs[2].plot(X_test, y_pred[:, 1], color='orange', label='Predicted Bench')
    axs[2].scatter(sample[2], sample[0], color='red', marker='x', s=100)
    axs[2].scatter(sample[2], sample[1], color='orange', marker='x', s=100)
    axs[2].set_xlabel('Deadlift Weight')
    axs[2].set_ylabel('Squat/Bench Weight')
    axs[2].set_title('Predictions from Deadlift Weight')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

# File path
file_path = './dataset/openpowerlifting.csv'  # Replace with your actual file path

# Load and prepare data
data = load_and_prepare_data(file_path)

# Train models
models = train_models(data)

# Sample data
sample = [150, 100, 200]  # Example sample values for Squat, Bench, and Deadlift

# Predict using sample data
squat_prediction = models['squat_to_bench_deadlift'].predict([[sample[0]]])
bench_prediction = models['bench_to_squat_deadlift'].predict([[sample[1]]])
deadlift_prediction = models['deadlift_to_squat_bench'].predict([[sample[2]]])

# Print predictions
print(f"Squat: {sample[0]}, Predicted Bench: {squat_prediction[0][0]}, Predicted Deadlift: {squat_prediction[0][1]}")
print(f"Bench: {sample[1]}, Predicted Squat: {bench_prediction[0][0]}, Predicted Deadlift: {bench_prediction[0][1]}")
print(f"Deadlift: {sample[2]}, Predicted Squat: {deadlift_prediction[0][0]}, Predicted Bench: {deadlift_prediction[0][1]}")

# Plot predictions with sample data
plot_predictions(data, models, sample)
