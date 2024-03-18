import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')  # Replace "your_dataset.csv" with the path to your dataset

# Separate features and target variable
X = data.drop(columns=['label'])
y = data['label']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
def fully_connected_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Compile the model
num_classes = len(np.unique(y))
input_shape = X_train_scaled.shape[1:]
model = fully_connected_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint_path = "model_weights_epoch_{epoch:02d}.h5"
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_freq='epoch')

class SaveBestModel(Callback):
    def __init__(self, filepath):
        super(SaveBestModel, self).__init__()
        self.filepath = filepath
        self.best_val_accuracy = -1

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.model.save_weights(self.filepath, overwrite=True)
            print(f"Model weights saved with validation accuracy: {val_accuracy:.4f}")

best_model_checkpoint = SaveBestModel(filepath="best_model_weights.h5")

# Train the model
history = model.fit(X_train_scaled, y_train_encoded, batch_size=16, epochs=200,
                    validation_data=(X_test_scaled, y_test_encoded),
                    callbacks=[checkpoint_callback, best_model_checkpoint])

# After training, load the weights of the best model based on validation accuracy
model.load_weights("best_model_weights.h5")

# Declare y_test here
y_test = label_encoder.inverse_transform(y_test_encoded)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Make predictions on the test data
predictions = model.predict(X_test_scaled)

# Convert predictions to classes
predicted_classes = np.argmax(predictions, axis=1)

# Convert predicted classes back to original labels
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy on Test Data: {accuracy*100}%")