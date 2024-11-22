import pandas as pd
import numpy as np

# Load your dataset
data = pd.read_csv('healthcare-dataset-stroke-data 2.csv')


# Define BMI categories
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi <= 24.9:
        return 'Normal weight'
    elif 25 <= bmi <= 29.9:
        return 'Overweight'
    elif bmi >= 30:
        return 'Obese'
    else:
        return 'Unknown'

# Define Age categories
def categorize_age(age):
    if age <= 18:
        return 'Child/Teen'
    elif 19 <= age <= 35:
        return 'Young Adult'
    elif 36 <= age <= 55:
        return 'Middle-aged Adult'
    elif age >= 56:
        return 'Senior'
    else:
        return 'Unknown'

def group_glucose_level(glucose):
    if glucose < 70:
        return 'Low'
    elif 70 <= glucose <= 140:
        return 'Normal'
    elif 141 <= glucose <= 199:
        return 'Prediabetic'
    else:
        return 'Diabetic'


data['BMI_Category'] = data['bmi'].apply(categorize_bmi)
data['Age_Group'] = data['age'].apply(categorize_age)
data['glucose_level_group'] = data['avg_glucose_level'].apply(group_glucose_level)


# Function to calculate entropy
def calculate_entropy(data, target_col):
    target_counts = data[target_col].value_counts()
    total = target_counts.sum()

    entropy = 0
    for count in target_counts:
        probability = count / total
        if probability > 0:  # Avoid log(0)
            entropy -= probability * np.log2(probability)
    return entropy


# Function to calculate information gain for a feature given a subset
def calculate_info_gain(data, target_col, feature_col, subset_col, subset_value):
    # Subset the data for the given subset value
    subset_data = data[data[subset_col] == subset_value]

    # Overall entropy of the target within the subset
    overall_entropy = calculate_entropy(subset_data, target_col)

    # Unique values of the feature
    unique_values = subset_data[feature_col].unique()

    # Weighted entropy calculation
    weighted_entropy = 0
    for value in unique_values:
        subset_value_data = subset_data[subset_data[feature_col] == value]
        weight = len(subset_value_data) / len(subset_data)
        entropy = calculate_entropy(subset_value_data, target_col)
        weighted_entropy += weight * entropy

    # Information Gain
    info_gain = overall_entropy - weighted_entropy
    return info_gain


# Define target, subset, and features to analyze
target_col = 'stroke'  # Target column
subset_col = 'gender'  # Subset column
subset_value = 'Female'  # Focus on Male
excluded_cols = ['id', 'gender', target_col]  # Exclude these columns
features = [col for col in data.columns if col not in excluded_cols]  # Relevant features

# Calculate Information Gain for each feature
print(f"Calculating Information Gain for features given {subset_col} = {subset_value}:\n")
for feature_col in features:
    info_gain = calculate_info_gain(data, target_col, feature_col, subset_col, subset_value)
    print(f"Feature: Male | {feature_col}, Information Gain: {info_gain:.4f}")


