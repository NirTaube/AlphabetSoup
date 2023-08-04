# AlphabetSoup
---
## Neural Network for Charity Success Predictor

This project involves the development of a binary classifier that is trained to predict the success of charitable projects based on several input features. The model uses a Deep Learning Neural Network and is implemented using TensorFlow and Keras.

## Dependencies

The project uses the following dependencies:

- Python 3.7
- TensorFlow
- Keras
- pandas
- scikit-learn

## Dataset

The dataset used in this project is `charity_data.csv`, which includes details of various charitable projects. The data is preprocessed by binning certain categorical variables and applying one-hot encoding. 

## Model Architecture

The model is a deep learning neural network with two different architectures:

1. A two-layered neural network: 
    - First layer with 8 nodes
    - Second layer with 5 nodes
    - Output layer with a sigmoid activation function

2. A three-layered neural network:
    - First layer with 16 nodes
    - Second layer with 8 nodes
    - Third layer with 4 nodes
    - Output layer with a sigmoid activation function

Additionally, a version of the model has been trained using early stopping and dropout layers for regularization.

## Training

The data is split into a training and testing set. The training data is then scaled using `StandardScaler` from scikit-learn. The model is trained for 100 epochs. 

A validation split of 0.1 is used for early stopping. The `patience` parameter for early stopping is set to 10.

## Evaluation

The model is evaluated using the testing data. The loss function used is binary cross-entropy, and the model metrics include accuracy.

## Results

The model's performance has been evaluated and it demonstrates reasonable accuracy.

## Usage

To train the model:
### relu / Sigmoid / Early Stopping
```python
# Define the model
nn = tf.keras.models.Sequential()
# Add layers
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim = input_features_total, activation = "relu"))
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation = "relu"))
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=100)
```

```python
# Define the model
input_features_total = len(X_train[0])
hidden_nodes_layer1 = 16
hidden_nodes_layer2 = 8
hidden_nodes_layer3 = 4

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim = input_features_total, activation = "relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation = "relu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation = "relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```

```python
#early stopping technique:
from tensorflow.keras.callbacks import EarlyStopping

# Define the model - deep neural net
input_features_total = len(X_train[0])
hidden_nodes_layer1 = 16
hidden_nodes_layer2 = 8
hidden_nodes_layer3 = 4

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim = input_features_total, activation = "relu"))
nn.add(tf.keras.layers.Dropout(0.2))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation = "relu"))
nn.add(tf.keras.layers.Dropout(0.2))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation = "relu"))
nn.add(tf.keras.layers.Dropout(0.2))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()

# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=100, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
