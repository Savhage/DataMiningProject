import pandas as pd
import numpy as np
from math import log2


def read_dataset(file_path):
    dataset = pd.read_csv(file_path)
    
    
    return dataset

def preprocessing(dataset):
    dataset=dataset.dropna()
    dataset=dataset[dataset['age']>5]
    dataset=dataset.sort_values(by='age')
    dataset['age_quartile'] = pd.qcut(dataset['age'], q=4, labels=[1, 2, 3, 4])
    dataset=dataset.sort_values(by='avg_glucose_level')
    dataset['avg_glucose_level_quartile'] = pd.qcut(dataset['avg_glucose_level'], q=4, labels=[1, 2, 3, 4])
    dataset=dataset.sort_values(by='bmi')
    dataset['bmi_quartile'] = pd.qcut(dataset['bmi'], q=4, labels=[1, 2, 3, 4])
    
    print('Age Quartiles')
    dataset=dataset.sort_values(by='age_quartile')
    quartile_ranges = pd.qcut(dataset['age'], q=4).unique()
    for i, q_range in enumerate(quartile_ranges, start=1):
        print(f"Quartile {i}: {q_range}")
    
    print('Glucose Quartile')
    dataset=dataset.sort_values(by='avg_glucose_level_quartile')
    quartile_ranges = pd.qcut(dataset['avg_glucose_level'], q=4).unique()
    for i, q_range in enumerate(quartile_ranges, start=1):
        print(f"Quartile {i}: {q_range}")
    
    print('BMI Quartile')
    dataset=dataset.sort_values(by='bmi_quartile')
    quartile_ranges = pd.qcut(dataset['bmi'], q=4).unique()
    for i, q_range in enumerate(quartile_ranges, start=1):
        print(f"Quartile {i}: {q_range}")

    dataset.drop(['id','age', 'avg_glucose_level', 'bmi', 'work_type'], axis=1, inplace=True)

def calculate_entropy(train_data, label, class_list):
    total = len(train_data)
    entropy = 0
    for cls in class_list:
        count = len(train_data[train_data[label] == cls])
        if count > 0:
            probability = count / total
            entropy -= probability * log2(probability)
    return entropy

def calculate_filtered_entropy(feature_value_data, label, class_list):
    total = len(feature_value_data)
    entropy = 0
    for cls in class_list:
        count = len(feature_value_data[feature_value_data[label] == cls])
        if count > 0:
            probability = count / total
            entropy -= probability * log2(probability)
    return entropy

def calculate_information_gain(train_data, feature_name, label, class_list):
    total_entropy = calculate_entropy(train_data, label, class_list)
    feature_values = train_data[feature_name].unique()
    weighted_entropy = 0
    total = len(train_data)
    
    for value in feature_values:
        subset = train_data[train_data[feature_name] == value]
        subset_entropy = calculate_filtered_entropy(subset, label, class_list)
        weighted_entropy += (len(subset) / total) * subset_entropy
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_best_feature(train_data, label, class_list):
    features = train_data.columns.drop(label)
    info_gains = {feature: calculate_information_gain(train_data, feature, label, class_list) for feature in features}
    best_feature = max(info_gains, key=info_gains.get)
    return best_feature

def find_best_feature_recursive(train_data, label, class_list, available_features=None, depth=0, min_info_gain=0.01):
    
    # Initialize available_features as all features except the label if not provided
    if available_features is None:
        available_features = [feature for feature in train_data.columns if feature != label]
    
    # Base case: Stop if no features are left or data is pure
    if not available_features or len(train_data[label].unique()) == 1:
        print(" " * depth + f"Leaf Node: {train_data[label].mode()[0]} (Pure class or no features left)")
        return

    # Calculate information gain for each available feature
    info_gains = {feature: calculate_information_gain(train_data, feature, label, class_list) for feature in available_features}

    # Find the best feature and its information gain
    best_feature = max(info_gains, key=info_gains.get)
    best_info_gain = info_gains[best_feature]

    # Stop if the information gain is below the threshold
    if best_info_gain < min_info_gain:
        print(" " * depth + f"Leaf Node: {train_data[label].mode()[0]} (Insufficient information gain)")
        return

    # Display the best feature and its information gain
    print(f"{' ' * depth}Best feature: {best_feature} (Information Gain: {best_info_gain:.4f})")

    # Remove the best feature from the available features list
    updated_available_features = [f for f in available_features if f != best_feature]

    # Simulate splitting the dataset and make recursive calls
    for value in train_data[best_feature].unique():
        subset = train_data[train_data[best_feature] == value]
        if subset.empty:
            continue

        # Display splitting condition
        print(f"{' ' * (depth + 2)}Splitting on {best_feature} = {value}")

        # Recursive call with the updated feature list
        find_best_feature_recursive(subset, label, class_list, updated_available_features, depth + 4, min_info_gain)

file_path = "healthcare-dataset-stroke-data 2.csv"
train_data = preprocessing(read_dataset(file_path))

label = "stroke"  # The label column
class_list = train_data['stroke'].unique()

# Generate the decision tree
print(calculate_entropy(train_data,label, class_list))
# Run the recursive function
find_best_feature_recursive(train_data, label, class_list)