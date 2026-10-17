# explore_titanic.py
# Introduction to Pandas with the Titanic dataset

import pandas as pd

# Step 1: Load the data
print("Loading Titanic data...")
df = pd.read_csv('data/titanic.csv')

# Step 2: Initial look
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Shape ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Step 3: Understand structure
print("\n--- Info ---")
df.info()

# Step 4: Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# Step 5: Categorical distributions
print("\n--- Sex Distribution ---")
print(df['Sex'].value_counts())

print("\n--- Survival Distribution ---")
print(df['Survived'].value_counts())

print("\n--- Passenger Class Distribution ---")
print(df['Pclass'].value_counts())

# Step 6: Answer analytical questions
print("\n--- Analytical Questions ---")

# Q1: How many total passengers?
total_passengers = df.shape[0]
print(f"Q1: Total passengers: {total_passengers}")

# Q2: What percentage survived?
survival_rate = df['Survived'].mean() * 100
print(f"Q2: Survival rate: {survival_rate:.1f}%")

# Q3: How many women in first class?
women_first_class = len(df[(df['Sex'] == 'female') & (df['Pclass'] == 1)])
print(f"Q3: Women in first class: {women_first_class}")

# Q4: What was the average fare?
avg_fare = df['Fare'].mean()
print(f"Q4: Average fare: ${avg_fare:.2f}")

# Q5: How many passengers under 18?
under_18 = len(df[df['Age'] < 18])
print(f"Q5: Passengers under 18: {under_18}")
