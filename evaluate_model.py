# evaluate_model.py
# Evaluate and improve the Titanic survival model

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Step 1: Load and prepare the data
print("Loading cleaned Titanic data...")
df = pd.read_csv('data/titanic_cleaned.csv')

X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2: Train the baseline model (same as Week 8)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nBaseline accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 3: Confusion matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Unpack the four values for clarity
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (predicted died, actually died): {tn}")
print(f"False Positives (predicted survived, actually died): {fp}")
print(f"False Negatives (predicted died, actually survived): {fn}")
print(f"True Positives (predicted survived, actually survived): {tp}")

# Step 4: Precision, recall, and F1
print("\n--- Precision, Recall, F1 ---")
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")

# Step 5: Cross-validation
print("\n--- Cross-Validation ---")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Scores for each fold: {cv_scores}")
print(f"Average accuracy: {cv_scores.mean():.2f}")

# Step 6: Feature importance (coefficients)
print("\n--- Feature Importance ---")
features = ['Pclass', 'Sex', 'Age']
coefficients = model.coef_[0]

for feature, coef in zip(features, coefficients):
    print(f"{feature}: {coef:.2f}")

# Step 7: Compare with a Decision Tree
print("\n--- Model Comparison ---")
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_pred)
print(f"Decision Tree accuracy: {tree_accuracy:.2f}")

print(f"\nLogistic Regression: {accuracy_score(y_test, y_pred):.2f}")
print(f"Decision Tree:       {tree_accuracy:.2f}")

# Step 8: Experiment with more features
print("\n--- Feature Experiment ---")

# Try adding FamilySize and IsAlone
X_extended = df[['Pclass', 'Sex', 'Age', 'FamilySize', 'IsAlone']]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_extended, y, test_size=0.2, random_state=42
)

model2 = LogisticRegression()
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)

print(f"3 features accuracy:  {accuracy_score(y_test, y_pred):.2f}")
print(f"5 features accuracy:  {accuracy_score(y_test2, y_pred2):.2f}")
