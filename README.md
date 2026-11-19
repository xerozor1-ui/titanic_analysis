# Titanic Data Analysis
An introduction to Pandas using the Titanic dataset.

# Overview
This project explores the Titanic dataset as part of the Python & AI Fundamentals course. It demonstrates loading data with Pandas, exploring data structure, and answering analytical questions.

# Dataset
- Source: Kaggle "Titanic: Machine Learning from Disaster" competition
- File: data/titanic.csv
- Rows: 891 passengers
- Columns: 12 attributes

# Questions Answered
1. How many total passengers were there?
2. What percentage survived?
3. How many women were in first class?
4. What was the average fare?
5. How many passengers were under 18?

# Course Progress
- Week 6: Load and explore data with Pandas
- Week 7: Data cleaning and feature engineering
- Week 8: First machine learning model (Logistic Regression)
- Week 9: Model evaluation and improvement
- Week 10: Deep learning (neural networks)

# Data Wrangling (Week 7)
- Filled missing Age with median
- Filled missing Embarked with mode
- Dropped Cabin (77% missing)
- Created FamilySize feature
- Created IsAlone feature
- Transformed Sex and Embarked to numerical
- Saved cleaned data to data/titanic_cleaned.csv
- Stored data in SQLite (titanic.db)

# Machine Learning (Week 8)
- Trained Logistic Regression model to predict survival
- Features: Pclass, Sex, Age
- Label: Survived
- Split: 80% training, 20% testing (random_state=42)
- Accuracy: 0.81 (baseline: 0.62)

# Model Evaluation (Week 9)
- Script: evaluate_model.py
- Confusion matrix: TN=91, FP=14, FN=20, TP=54 (dominant error: false negatives)
- Precision: 0.79, Recall: 0.73, F1: 0.76
- 5-fold cross-validation average accuracy: 0.79
- Coefficients: Sex 2.47, Pclass -1.04, Age -0.03 (Sex is the strongest predictor)
- Logistic Regression 0.81 vs. Decision Tree 0.78
- Feature experiment: adding FamilySize and IsAlone gave 0.80 vs. 0.81 with 3 features (no improvement)

# Deep Learning (Week 10)
- Script: neural_network.py (guided tour, annotated with explanations of each step)
- TensorFlow/Keras Sequential model: 3 inputs -> Dense(16, relu) -> Dense(8, relu) -> Dense(1, sigmoid)
- Features scaled with StandardScaler; compiled with adam + binary_crossentropy; trained 50 epochs, batch size 32
- Neural network accuracy: ~0.79-0.81 (varies run to run due to random weight initialization)
- Modification: added a Dense(4, relu) hidden layer -> mean 0.81 over 3 runs vs. 0.79 original (within run-to-run noise)
- Comparison: Logistic Regression 0.81 vs. neural network ~0.79-0.81, essentially a tie on this small tabular dataset
