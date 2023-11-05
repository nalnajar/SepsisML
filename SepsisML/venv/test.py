import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the data from "sepsis_survival_primary_cohort.csv"
load_data = pd.read_csv('csv/sepsis_survival_primary_cohort.csv')

# Select features and the target variable
features = load_data[['age_years', 'sex_0male_1female', 'episode_number']]
target = load_data['hospital_outcome_1alive_0dead']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2)

# Create a decision tree regressor
regressor = DecisionTreeRegressor()

# Fit the model on the training data
regressor.fit(X_train, y_train)

# Get user input for age, sex, and episode number
age = float(input("Enter age: "))
sex = int(input("Enter sex (0 for male, 1 for female): "))
episode_number = int(input("Enter episode number: "))

# Number of days untreated (you can adjust this as needed)
days_untreated = int(input("Enter number of days untreated: "))

# Calculate the daily increased risk based on your range (4-9% per hour)
daily_increased_risk_range = (0.04, 0.09)
daily_increased_risk_per_day = sum(
    daily_increased_risk_range) / 2  # Average risk per hour

# Prepare the user's input as a feature vector
user_input = [[age, sex, episode_number]]

# Make predictions for the user's input
immediate_prediction = regressor.predict(user_input)

# Predicting after 9 days (you may need to adjust this)
episode_number_after_9_days = episode_number + 9
user_input_after_9_days = [[age, sex, episode_number_after_9_days]]
prediction_after_9_days = regressor.predict(user_input_after_9_days)

# Interpret the predictions


def interpret_prediction(prediction):
    if prediction > 0.5:
        return "alive"
    else:
        return "dead"


immediate_result = interpret_prediction(immediate_prediction)
result_after_9_days = interpret_prediction(prediction_after_9_days)

# Calculate the odds of dying within 9 days
average_survival_rate = 0.30  # 30% survival rate

if immediate_prediction == 0:
    odds_of_dying_immediately = 1.0  # Certain death if immediate prediction is "dead"
else:
    odds_of_dying_immediately = 1 - \
        (1 - average_survival_rate) ** days_untreated

if prediction_after_9_days == 0:
    # Certain death if prediction after 9 days is "dead"
    odds_of_dying_after_9_days = 1.0
else:
    odds_of_dying_after_9_days = 1 - \
        (1 - average_survival_rate) ** (9 + days_untreated)

# Print the predictions and odds to the screen
print("Predicting hospital outcome based on the provided data:")
print(f"Age: {age}")
print(f"Sex (0 for male, 1 for female): {sex}")
print(f"Episode number: {episode_number}")
print(f"Number of days untreated: {days_untreated} days")
print(
    f"Based on the provided data, the individual is predicted to be {immediate_result} immediately.\n")
print(
    f"Odds of dying after {days_untreated} days: {odds_of_dying_immediately:.2%}")
print(
    f"Odds of dying within 9 days if left untreated: {odds_of_dying_after_9_days:.2%}\n")

# Calculate Mean Squared Error and R-squared on the test data
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Count the number of alive and dead individuals in the test data
alive_count_test = sum(y_test)
dead_count_test = len(y_test) - alive_count_test

# Print the counts of alive and dead individuals in test and training data
print(f"Number of individuals in the test data: {len(y_test)}")
print(
    f"Number of individuals predicted as alive (test data): {alive_count_test}")
print(
    f"Number of individuals predicted as dead (test data): {dead_count_test}")

# Calculate the number of episodes for each individual in the test data
episode_counts = X_test['episode_number'].values

# Create a dictionary to count the number of deaths for each episode count in the test data
deaths_by_episode_test = {}
for episode_count, outcome in zip(episode_counts, y_test):
    if outcome < 0.5:
        if episode_count in deaths_by_episode_test:
            deaths_by_episode_test[episode_count] += 1
        else:
            deaths_by_episode_test[episode_count] = 1

# Print the count of deaths by episode in the test data
print("\nNumber of Deaths by Episode Count (Test Data):")
for episode_count, death_count in deaths_by_episode_test.items():
    print(f"Episode Count: {episode_count} - Deaths: {death_count}")

# Calculate the number of episodes for each individual in the training data
episode_counts = X_train['episode_number'].values

# Create a dictionary to count the number of deaths for each episode count in the training data
deaths_by_episode_train = {}
for episode_count, outcome in zip(episode_counts, y_train):
    if outcome < 0.5:
        if episode_count in deaths_by_episode_train:
            deaths_by_episode_train[episode_count] += 1
        else:
            deaths_by_episode_train[episode_count] = 1

# Print the count of deaths by episode in the training data
print("\nNumber of Deaths by Episode Count (Training Data):")
for episode_count, death_count in deaths_by_episode_train.items():
    print(f"Episode Count: {episode_count} - Deaths: {death_count}")

# Visualize the decision tree
plt.figure(figsize=(10, 6))
plot_tree(regressor, feature_names=features.columns,
          filled=True, rounded=True, fontsize=6)

plt.show()


# Create a new DataFrame for the test data
test_data = X_test.copy()
test_data['hospital_outcome'] = y_test

# Group the test data by age and count the number of deaths for each age group
age_death_counts = test_data.groupby('age_years')['hospital_outcome'].sum()

# Plot the age vs. death counts
plt.figure(figsize=(10, 6))
plt.bar(age_death_counts.index, age_death_counts.values, width=0.5)
plt.xlabel('Age (years)')
plt.ylabel('Number of Deaths')
plt.title('Age vs. Number of Deaths')
plt.show()
