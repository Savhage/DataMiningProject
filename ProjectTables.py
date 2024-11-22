import pandas as pd
import numpy as np
from math import log2
import matplotlib.pyplot as plt


def readDataset(filePath):
    dataset = pd.read_csv(filePath)
    return dataset


def preprocessing(dataset):
    dataset = dataset.dropna()
    dataset = dataset[dataset['age'] > 1]
    dataset = dataset.sort_values(by='age')
    dataset['ageQuartile'] = pd.qcut(dataset['age'], q=4, labels=[1, 2, 3, 4])
    dataset = dataset.sort_values(by='avg_glucose_level')
    dataset['avgGlucoseLevelQuartile'] = pd.qcut(dataset['avg_glucose_level'], q=4, labels=[1, 2, 3, 4])
    dataset = dataset.sort_values(by='bmi')
    dataset['bmiQuartile'] = pd.qcut(dataset['bmi'], q=4, labels=[1, 2, 3, 4])

    print('Age Quartiles')
    quartileRanges = pd.qcut(dataset['age'], q=4).unique()
    for i, qRange in enumerate(quartileRanges, start=1):
        print(f"Quartile {i}: {qRange}")

    print('Glucose Quartiles')
    quartileRanges = pd.qcut(dataset['avg_glucose_level'], q=4).unique()
    for i, qRange in enumerate(quartileRanges, start=1):
        print(f"Quartile {i}: {qRange}")

    print('BMI Quartiles')
    quartileRanges = pd.qcut(dataset['bmi'], q=4).unique()
    for i, qRange in enumerate(quartileRanges, start=1):
        print(f"Quartile {i}: {qRange}")

    dataset.drop(['id', 'work_type'], axis=1, inplace=True)
    return dataset


def calculateEntropy(trainData, label, classList):
    total = len(trainData)
    entropy = 0
    for cls in classList:
        count = len(trainData[trainData[label] == cls])
        if count > 0:
            probability = count / total
            entropy -= probability * log2(probability)
    return entropy


def calculateStrokeLikelihood(dataset, columnName, strokeColumn,datacolumn,datatarget):
    """
    Calculates and plots stroke likelihood based on a specified column (e.g., age, gender).

    Parameters:
        dataset (pd.DataFrame): The dataset containing the specified column and stroke data.
        columnName (str): The column name to group data by (e.g., 'age', 'gender').
        strokeColumn (str): The column name for stroke (1 = stroke, 0 = no stroke).
    """
    # Use quartile columns if the column is age, bmi, or avg_glucose_level
    if columnName in ['age', 'bmi']:
        columnName = f'{columnName}Quartile'
    if columnName in ['avg_glucose_level']:
        columnName = 'avgGlucoseLevelQuartile'

    # Group data by the specified column and calculate stroke likelihood
    strokeLikelihood = dataset.groupby(columnName)[strokeColumn].mean()

    # Print likelihood details
    print(f"\nStroke Likelihood by {columnName.title()}:")
    for group, likelihood in strokeLikelihood.items():
        print(f"{columnName.title()}: {group}, Stroke Likelihood: {likelihood:.4f}")

    # Plot the results
    print(f"\nPlotting Stroke Likelihood For {datacolumn} {datatarget} per {columnName}...")
    plt.figure(figsize=(10, 6))
    strokeLikelihood.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Stroke Likelihood For {datacolumn}: {datatarget} and {columnName}', fontsize=16)
    plt.xlabel(columnName.title(), fontsize=14)
    plt.ylabel('Stroke Likelihood', fontsize=14)
    plt.xticks(rotation=45 if 'Quartile' in columnName else 0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return strokeLikelihood


filePath = "healthcare-dataset-stroke-data 2.csv"
trainData = preprocessing(readDataset(filePath))

label = "stroke"  # The label column
classList = trainData['stroke'].unique()
print(calculateEntropy(trainData, label, classList))

rawData = readDataset(filePath)

# Preprocess the dataset
processedData = preprocessing(rawData)

column = 'bmiQuartile'
target = 1

processedData = processedData[processedData[column] == target]

# Calculate and plot stroke likelihood by column name
calculateStrokeLikelihood(processedData, 'avg_glucose_level', 'stroke',column,target)
