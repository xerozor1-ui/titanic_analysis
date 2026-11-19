# build_nn.py
# Build, evaluate, and compare a neural network for Titanic survival (Week 11)
#
# =====================================================================
# PSEUDOCODE (written before the code)
# =====================================================================
# IMPORT pandas, train_test_split, cross_val_score, StandardScaler
# IMPORT LogisticRegression, make_pipeline
# IMPORT accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
# IMPORT Sequential, Dense, EarlyStopping from tensorflow.keras
#
# Step 1: Load and prepare
#   LOAD data/titanic_cleaned.csv into df
#   SET X = df[Pclass, Sex, Age]   (3 features)
#   SET y = df[Survived]           (label)
#   SPLIT X, y into 80% train / 20% test (random_state=42)
#
# Step 2: Scale features
#   FIT StandardScaler on the TRAINING data only
#   TRANSFORM both training and test data with that scaler
#
# Step 3: Build the network
#   CREATE Sequential model
#   ADD Dense(16, relu) with input_shape=(3,)   <- hidden layer 1
#   ADD Dense(8, relu)                          <- hidden layer 2
#   ADD Dense(1, sigmoid)                       <- output: survival probability
#
# Step 4: Compile
#   COMPILE with optimizer=adam, loss=binary_crossentropy, metrics=accuracy
#
# Step 5: Train
#   FIT on scaled training data, epochs=50, batch_size=32, silent
#
# Step 6: Predict and evaluate
#   PREDICT probabilities on scaled test data
#   CONVERT to 0/1 with threshold 0.5, FLATTEN to 1D
#   CALCULATE and PRINT accuracy
#
# Step 7: Full evaluation (Week 9 toolkit)
#   PRINT confusion matrix, UNPACK TN/FP/FN/TP
#   PRINT precision, recall, F1
#
# Step 8: Fair comparison
#   TRAIN Logistic Regression on the SAME scaled split, PRINT test accuracy
#   RUN 5-fold CV on a scaler+LR pipeline, PRINT mean accuracy
#
# Step 9: Early stopping
#   SPLIT a validation set out of the TRAINING data (test set stays untouched)
#   BUILD + COMPILE a fresh copy of the same network
#   FIT up to 200 epochs, stop when val_loss hasn't improved for 10 epochs,
#     restore the best weights
#   PRINT test accuracy and number of epochs actually trained
#
# Comparison to the provided design: my plan has the same six core steps in
# the same order, including scaling (fit on train only). It adds Steps 7-9.
# =====================================================================

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load and prepare the data
df = pd.read_csv('data/titanic_cleaned.csv')
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2: Scale the features (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Build the neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 4: Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the model
model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    verbose=0
)

# Step 6: Predict and evaluate
probs = model.predict(X_test_scaled, verbose=0)
y_pred_nn = (probs > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred_nn)
print(f"Neural network accuracy: {accuracy:.2f}")

# Step 7: Full evaluation with Week 9's toolkit
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred_nn)
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

precision = precision_score(y_test, y_pred_nn)
recall = recall_score(y_test, y_pred_nn)
f1 = f1_score(y_test, y_pred_nn)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")

# Step 8: Compare with Logistic Regression (fair comparison)
print("\n--- Model Comparison ---")

# Same split, same scaled data
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"Logistic Regression test accuracy: {lr_accuracy:.2f}")
print(f"Neural network test accuracy:      {accuracy:.2f}")

# More trustworthy LR estimate with cross-validation
lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression())
lr_cv_scores = cross_val_score(lr_pipeline, X, y, cv=5)
print(f"Logistic Regression CV accuracy:   {lr_cv_scores.mean():.2f}")

# Step 9: Early stopping experiment
print("\n--- Early Stopping ---")

# Carve a validation set out of the training data
X_train2, X_val, y_train2, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model2 = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model2.fit(
    X_train2, y_train2,
    epochs=200,
    batch_size=32,
    verbose=0,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

probs2 = model2.predict(X_test_scaled, verbose=0)
y_pred2 = (probs2 > 0.5).astype(int).flatten()
acc2 = accuracy_score(y_test, y_pred2)

print(f"Early-stopped NN accuracy: {acc2:.2f}")
print(f"Epochs trained: {len(history.history['loss'])}")

# =====================================================================
# Hyperparameter experiment results (one change at a time, 3 runs each,
# same train/test split; accuracy varies ~0.77-0.81 run to run)
# =====================================================================
# Baseline (16-8 neurons, 2 hidden layers, 50 epochs, batch 32):
#   accuracy 0.81, 0.79, 0.77 (mean 0.79)   recall 0.64, 0.65, 0.66
# Experiment: L2 regularization, kernel_regularizer=l2(0.01) on both hidden
#   layers (from tensorflow.keras.regularizers import l2):
#   accuracy 0.79, 0.82, 0.79 (mean 0.80)   recall 0.70, 0.73, 0.73
# Conclusion: accuracy did not meaningfully change (+0.01 is within the noise),
#   but L2 consistently raised recall (~0.65 -> ~0.72), i.e. fewer false
#   negatives (missed survivors). Penalizing large weights kept the network
#   from leaning so heavily on predicting "died". Not a clear enough
#   accuracy win to keep; reverted to baseline.
#
# Other options checked for context (mean accuracy, 3 runs):
#   32 neurons 0.80 | extra Dense(4) layer 0.80 | 100 epochs 0.80 |
#   batch_size=16 0.80 -> none beat the noise on this small dataset.
#
# Other observations:
#   - Dominant error is false negatives (e.g. FN=20-33 vs FP=8-17), so
#     precision (0.75-0.84) > recall (0.55-0.73): the network is cautious
#     about predicting survival, same pattern as Week 9 Logistic Regression
#     (precision 0.79, recall 0.73).
#   - Fair comparison: LR on scaled data 0.80 test / 0.79 CV vs NN 0.77-0.79.
#     Tie or slight LR edge.
#   - Early stopping halted at 40-50 epochs (out of 200) with test accuracy
#     0.77-0.81: same accuracy, far less training.
