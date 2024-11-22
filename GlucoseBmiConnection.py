import pandas as pd
import numpy as np

data = pd.read_csv('healthcare-dataset-stroke-data 2.csv')


def group_glucose_level(glucose):
    if glucose < 70:
        return 'Low'
    elif 70 <= glucose <= 140:
        return 'Normal'
    elif 141 <= glucose <= 199:
        return 'Prediabetic'
    else:
        return 'Diabetic'

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