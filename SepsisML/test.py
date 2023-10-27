import pandas as pd

# Load the data from "sepsis_survival_primary_cohort.csv"
load_data = pd.read_csv('csv/sepsis_survival_primary_cohort.csv')

features = load_data[['age_years', 'sex_0male_1female', 'episode_number']]

# Print the first few rows of the split data
print("Features (X):")
print(features.head())

target = load_data['hospital_outcome_1alive_0dead']

print("\nTarget Variable (Y):")
print(target.head())
