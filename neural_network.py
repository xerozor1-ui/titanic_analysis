# neural_network.py
# A neural network for Titanic survival prediction (guided tour)
#
# =====================================================================
# WEEK 10 ANNOTATIONS
# =====================================================================
#
# --- Imports (Step 7) ---
# StandardScaler: rescales each feature to mean 0 and standard deviation 1.
#   A neural network's weights are updated by gradient descent, and features on
#   very different scales (Age ~0-80 vs. Sex 0/1) make that slow and unstable.
#   Scaling puts every feature on equal footing so training converges well.
# Sequential: builds a model as a simple linear stack of layers, where the
#   output of each layer feeds straight into the next one.
# Dense: a fully connected layer. "Dense" means every neuron in the layer
#   is connected to every output of the previous layer.
#
# --- Model architecture (Step 8) ---
# Hidden layers: 2 in the original (16 neurons, then 8). My modification adds a
#   third hidden layer of 4 neurons (see Part IV below).
# Output layer: 1 neuron, because this is binary classification (survived or
#   not). One neuron outputs one number: the probability of survival.
# input_shape=(3,): only the first layer needs to be told the input size (our
#   3 features: Pclass, Sex, Age). Each later layer infers its input size from
#   the number of neurons in the layer before it.
# relu vs. sigmoid: ReLU (max(0, x)) in the hidden layers adds non-linearity,
#   so the network can learn curved patterns. It is also cheap to compute and
#   avoids vanishing gradients. Sigmoid in the output squashes the result into
#   0-1 so it can be read as a probability, exactly like Logistic Regression.
#   (Logistic Regression = a single neuron with a sigmoid activation.)
#
# --- compile() (Step 9) ---
# Loss function: measures how wrong the model's predictions are, and training
#   tries to minimize it. binary_crossentropy is the standard loss for a
#   two-class problem with a sigmoid output. It heavily penalizes confident
#   wrong probabilities (same log-loss used by Logistic Regression).
# Optimizer: the algorithm that adjusts the weights to reduce the loss. Adam
#   is a gradient-descent variant that adapts the learning rate for each weight
#   automatically, so it works well without much tuning.
# metrics=['accuracy']: loss is what the model optimizes, but it is hard for a
#   human to interpret. Accuracy is reported alongside it so we can follow
#   progress in plain terms (fraction of correct predictions).
#
# --- fit() (Step 10) ---
# Epoch: one full pass through all the training data. epochs=50 means the
#   model sees every training passenger 50 times.
# Batch size: the number of samples processed before the weights are updated.
#   batch_size=32 means weights update after every 32 passengers (712 training
#   rows / 32 = ~23 updates per epoch, ~1,150 updates over 50 epochs).
# X_train_scaled: the model must train on scaled data (see StandardScaler).
#   The scaler is fit on training data only, then the same transform is
#   applied to the test data, so no information leaks from the test set.
# verbose=0: silent training. verbose=1 prints a progress bar with the loss
#   and accuracy for each of the 50 epochs.
#
# --- Prediction (Step 11) ---
# model.predict() returns probabilities, not class labels. You can tell
#   because the values are decimals between 0 and 1 (from the sigmoid
#   output), with shape (n_samples, 1).
# (probs > 0.5): turns each probability into True/False. 0.5 is the natural
#   decision boundary: predict "survived" if the model thinks survival is more
#   likely than not. Same default threshold as Logistic Regression.
# .astype(int): converts True/False into 1/0 to match the y_test labels.
# .flatten(): turns the 2D (n, 1) array into a 1D (n,) array, the same
#   shape as y_test, which accuracy_score expects.
# accuracy_score(): the same function from Weeks 8-9. Evaluation does not care
#   whether predictions came from a neural network or classical ML: same
#   train/test split, same metrics, same comparison to a baseline.
#
# --- Part IV: Modification ---
# Change made: added a third hidden layer, Dense(4, activation='relu'),
#   before the output layer.
# Results (3 runs each, same train/test split):
#   Original (16-8-1, 50 epochs):  0.78, 0.78, 0.79  (mean 0.79)
#   + Dense(4) layer (16-8-4-1):   0.82, 0.83, 0.79  (mean 0.81)
#   For reference, other options:  32 neurons mean 0.80, 100 epochs mean 0.80,
#                                  10 epochs mean 0.79
# Observation: the extra layer scored slightly higher on average, but the
#   difference (~0.02) is about the same size as the run-to-run noise from
#   random weight initialization. Depth did not clearly help. With only 3
#   features and ~700 training rows, there isn't much extra pattern for more
#   capacity to find.
#
# --- Comparison to Logistic Regression ---
# Logistic Regression (Week 8, same split): 0.81 (5-fold CV average: 0.79).
# Neural network: ~0.79-0.81 depending on the run.
# Verdict: essentially a tie. The neural network does not beat Logistic
#   Regression on this small, tabular, 3-feature problem, and it is slower,
#   less interpretable, and non-deterministic.
# =====================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load and prepare the data
df = pd.read_csv('data/titanic_cleaned.csv')
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Build the neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),  # Part IV modification: added layer
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

# Step 7: Compare to the Logistic Regression baseline
print(f"Neural network accuracy: {accuracy:.2f}")
print(f"Logistic Regression accuracy (from Week 8): 0.81 (5-fold CV avg: 0.79)")
