# predict_survival.py
# Train a Logistic Regression model to predict Titanic survival

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the cleaned data
print("Loading cleaned Titanic data...")
df = pd.read_csv('data/titanic_cleaned.csv')
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# Step 2: Define features (X) and labels (y)
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

print(f"\nFeatures: {X.shape[1]} columns")
print(f"Labels: {y.shape[0]} values")

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} rows")
print(f"Testing set: {X_test.shape[0]} rows")

# Step 4: Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

print("\nModel trained.")

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

print(f"\nPredictions made for {len(y_pred)} test passengers.")
print(f"First 10 predictions: {y_pred[:10]}")

# Step 6: Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Step 7: Compare to a baseline
baseline = 1 - y.mean()
print(f"\nBaseline (predict everyone died): {baseline:.2f}")
print(f"Model improvement over baseline: {accuracy - baseline:.2f}")
