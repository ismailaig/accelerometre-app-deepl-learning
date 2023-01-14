import pandas as pd
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import lite


# Load the data into a pandas dataframe
df = pd.read_csv("WISDM_ar_v1.1_raw.txt", delimiter=",", on_bad_lines='skip', header=None, names=["user_id", "activity", "timestamp", "x", "y", "z"])

# Get some basic statistics about the data
print(df.shape)
print(df.describe())
print(df["activity"].value_counts())

# Handle missing values
df = df.dropna()

# Normalize the features
df["z"] = df["z"].str.replace(';','')
scaler = MinMaxScaler()
df[["x", "y", "z"]] = scaler.fit_transform(df[["x", "y", "z"]])

# Encode the labels
encoder = LabelEncoder()
df["activity"] = encoder.fit_transform(df["activity"])

# Extract the sensor data and activity labels
X = df[['x', 'y', 'z']].values
y = df['activity'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(X_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = keras.Sequential()
model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.MaxPool1D(pool_size=1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(len(encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', test_acc)

# Saving the model in HDF5 format
model.save("activity_recog.h5")

# Loading the model from HDF5 file
model = keras.models.load_model("activity_recog.h5")

# Converting the model to TensorFlow Lite format
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Converting a SavedModel to a TensorFlow Lite model.
open("activity_recog.tflite", "wb").write(tflite_model)