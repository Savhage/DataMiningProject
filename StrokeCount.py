import pandas as pd

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



# Function to count strokes (0 and 1) for a subset and grouped by another attribute
def count_strokes_by_attribute(data, subset_col, subset_value, group_col, target_col):
    # Subset the data for the given subset value
    subset_data = data[data[subset_col] == subset_value]

    # Group by the specified attribute and count the strokes
    stroke_counts = subset_data.groupby(group_col)[target_col].value_counts().unstack(fill_value=0)

    # Rename columns for clarity
    stroke_counts.columns = ['stroke=0', 'stroke=1']

    return stroke_counts


# Define subset and group parameters
subset_col = 'gender'  # Subset column
subset_value = 'Male'  # Focus on Male
group_col = 'smoking_status'  # Attribute to group by
target_col = 'stroke'  # Target column

if group_col=='smoking_status':
    data = data[data['smoking_status'] != 'Unknown']


# Calculate and print stroke counts
stroke_counts = count_strokes_by_attribute(data, subset_col, subset_value, group_col, target_col)
print(f"Stroke counts for {subset_col} = {subset_value} grouped by {group_col}:\n")
print(stroke_counts)


