import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the data from "sepsis_survival_primary_cohort.csv"
# load_data = pd.read_csv('csv/sepsis_survival_primary_cohort.csv')
# load_data = pd.read_csv('csv/sepsis_survival_study_cohort.csv')
load_data = pd.read_csv('csv/sepsis_survival_validation_cohort.csv')


# Select features and target variable
features = load_data[['age_years', 'sex_0male_1female', 'episode_number']]
target = load_data['hospital_outcome_1alive_0dead']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)

# Create a decision tree regressor
regressor = DecisionTreeRegressor(random_state=42)

# Fit the model on the training data
regressor.fit(X_train, y_train)

# Get user input for age, sex, and episode number
age = float(input("Enter age: "))
sex = int(input("Enter sex (0 for male, 1 for female): "))
episode_number = int(input("Enter episode number: "))

# Prepare the user's input as a feature vector
user_input = [[age, sex, episode_number]]

# Make a prediction for the user's input
prediction = regressor.predict(user_input)

print(f"prediction: {prediction}")

# Interpret the prediction
if prediction > 0.5:
    result = "alive"
else:
    result = "dead"

# Print the prediction to the screen
print("Predicting hospital outcome based on the provided data:")
print(f"Age: {age}")
print(f"Sex (0 for male, 1 for female): {sex}")
print(f"Episode number: {episode_number}")
print(
    f"Based on the provided data, the individual is predicted to be {result}.")

# Calculate Mean Squared Error and R-squared on the test data
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualize the decision tree
plt.figure(figsize=(10, 6))
plot_tree(regressor, feature_names=features.columns, filled=True, rounded=True)
plt.show()
