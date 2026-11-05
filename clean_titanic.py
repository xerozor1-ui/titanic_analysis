# clean_titanic.py
# Data wrangling for the Titanic dataset

import pandas as pd
import sqlite3

# Step 1: Load the data
print("Loading Titanic data...")
df = pd.read_csv('data/titanic.csv')
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# Step 2: Detect missing values
print("\n--- Missing Values ---")
missing = df.isna().sum()
print(missing[missing > 0])

# Step 3: Clean the data
print("\n--- Cleaning Data ---")

# Fill missing ages with the median age
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"Filled missing Age values with median: {median_age}")

# Fill missing embarkation ports with the mode (most common)
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)
print(f"Filled missing Embarked values with mode: {mode_embarked}")

# Drop the Cabin column (77% missing - too much to fill)
df = df.drop('Cabin', axis=1)
print("Dropped Cabin column (too much missing data).")

# Verify no missing values remain
remaining_missing = df.isna().sum().sum()
print(f"\nTotal remaining missing values: {remaining_missing}")

# Step 4: Engineer new features
print("\n--- Feature Engineering ---")

# FamilySize = siblings/spouses + parents/children + the passenger
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("Created FamilySize column.")

# IsAlone = 1 if traveling alone, 0 if traveling with family
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print("Created IsAlone column (1 = alone, 0 = with family).")

# Step 5: Transform categorical columns
print("\n--- Transforming Data ---")

# Map Sex to numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print("Mapped Sex: male -> 0, female -> 1")

# Map Embarked to numerical values
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
print("Mapped Embarked: C -> 0, Q -> 1, S -> 2")

# Step 6: Sort and answer questions
print("\n--- Sorting Questions ---")

# Who paid the most?
print("\nTop 5 most expensive tickets:")
print(df.sort_values('Fare', ascending=False).head()[['Name', 'Fare']])

# Who were the oldest passengers?
print("\nTop 5 oldest passengers:")
print(df.sort_values('Age', ascending=False).head()[['Name', 'Age']])

# Step 7: Save the cleaned data
print("\n--- Saving Data ---")

df.to_csv('data/titanic_cleaned.csv', index=False)
print("Saved cleaned data to data/titanic_cleaned.csv")

# Step 8: Store in SQLite and query
print("\n--- SQLite Storage ---")

# Connect to (or create) the database
conn = sqlite3.connect('titanic.db')
print("Connected to titanic.db.")

# Write the DataFrame to a table
df.to_sql('titanic', conn, if_exists='replace', index=False)
print("Wrote DataFrame to 'titanic' table.")

# Query: How many survivors?
query = "SELECT COUNT(*) FROM titanic WHERE Survived = 1"
survivor_count = pd.read_sql_query(query, conn).iloc[0, 0]
print(f"Survivors: {survivor_count}")

# Query: Female first-class survivors
query = """
    SELECT COUNT(*) FROM titanic
    WHERE Survived = 1
    AND Sex = 1
    AND Pclass = 1
"""
female_first_class_survivors = pd.read_sql_query(query, conn).iloc[0, 0]
print(f"Female first-class survivors: {female_first_class_survivors}")

# Close the connection
conn.close()
print("Database connection closed.")
