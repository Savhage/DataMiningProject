import pandas as pd
import numpy as np
from math import log2
import matplotlib.pyplot as plt


def read_dataset(file_path):
    dataset = pd.read_csv(file_path)
    
    
    return dataset

def preprocessing(dataset):
    dataset = dataset.dropna()
    dataset = dataset[dataset['age'] > 1]
    dataset = dataset.sort_values(by='age')
    dataset['age_quartile'] = pd.qcut(dataset['age'], q=4, labels=[1, 2, 3, 4])
    dataset = dataset.sort_values(by='avg_glucose_level')
    dataset['avg_glucose_level_quartile'] = pd.qcut(dataset['avg_glucose_level'], q=4, labels=[1, 2, 3, 4])
    dataset = dataset.sort_values(by='bmi')
    dataset['bmi_quartile'] = pd.qcut(dataset['bmi'], q=4, labels=[1, 2, 3, 4])
    
    # Print quartile ranges for debugging
    print('Age Quartiles')
    quartile_ranges = pd.qcut(dataset['age'], q=4).unique()
    for i, q_range in enumerate(quartile_ranges, start=1):
        print(f"Quartile {i}: {q_range}")
    
    print('Glucose Quartiles')
    quartile_ranges = pd.qcut(dataset['avg_glucose_level'], q=4).unique()
    for i, q_range in enumerate(quartile_ranges, start=1):
        print(f"Quartile {i}: {q_range}")
    
    print('BMI Quartiles')
    quartile_ranges = pd.qcut(dataset['bmi'], q=4).unique()
    for i, q_range in enumerate(quartile_ranges, start=1):
        print(f"Quartile {i}: {q_range}")
    
    # Remove unnecessary columns but retain original columns used for quartiles
    dataset.drop(['id', 'work_type'], axis=1, inplace=True)
    return dataset

def calculate_entropy(train_data, label, class_list):
    total = len(train_data)
    entropy = 0
    for cls in class_list:
        count = len(train_data[train_data[label] == cls])
        if count > 0:
            probability = count / total
            entropy -= probability * log2(probability)
    return entropy


def calculate_stroke_likelihood(dataset, column_name, stroke_column):
    """
    Calculates and plots stroke likelihood based on a specified column (e.g., age, gender).

    Parameters:
        dataset (pd.DataFrame): The dataset containing the specified column and stroke data.
        column_name (str): The column name to group data by (e.g., 'age', 'gender').
        stroke_column (str): The column name for stroke (1 = stroke, 0 = no stroke).
    """
    # Use quartile columns if the column is age, bmi, or avg_glucose_level
    if column_name in ['age', 'bmi', 'avg_glucose_level']:
        column_name = f'{column_name}_quartile'

    # Group data by the specified column and calculate stroke likelihood
    stroke_likelihood = dataset.groupby(column_name)[stroke_column].mean()

    # Print likelihood details
    print(f"\nStroke Likelihood by {column_name.title()}:")
    for group, likelihood in stroke_likelihood.items():
        print(f"{column_name.title()}: {group}, Stroke Likelihood: {likelihood:.4f}")

    # Plot the results
    print(f"\nPlotting Stroke Likelihood by {column_name.title()}...")
    plt.figure(figsize=(10, 6))
    stroke_likelihood.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Stroke Likelihood by {column_name.title()}', fontsize=16)
    plt.xlabel(column_name.title(), fontsize=14)
    plt.ylabel('Stroke Likelihood', fontsize=14)
    plt.xticks(rotation=45 if 'quartile' in column_name else 0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return stroke_likelihood


file_path = "healthcare-dataset-stroke-data 2.csv"
train_data = preprocessing(read_dataset(file_path))

label = "stroke"  # The label column
class_list = train_data['stroke'].unique()
print(calculate_entropy(train_data,label, class_list))

file_path = "healthcare-dataset-stroke-data 2.csv"
raw_data = read_dataset(file_path)

# Preprocess the dataset
processed_data = preprocessing(raw_data)

processed_data=processed_data[processed_data["avg_glucose_level_quartile"]==3]
# Calculate and plot stroke likelihood by column name
calculate_stroke_likelihood(processed_data, 'age', 'stroke')