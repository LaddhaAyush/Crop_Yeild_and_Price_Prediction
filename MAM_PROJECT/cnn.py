import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Display dataset information
print(df.info())

# Extract features and labels
all_columns = df.columns[:-1]
label_encoder = LabelEncoder()
X = df[all_columns]
y = label_encoder.fit_transform(df["label"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)

# SVM model
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=0))
svm_pipeline.fit(X_train, y_train)

# Save the SVM model
pickle.dump(svm_pipeline, open("svm_pipeline.pkl", "wb"))

# Neural Network model
num_classes = len(np.unique(y))
input_shape = X_train.shape[1:]
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the neural network model
history = model.fit(X_train, y_train, batch_size=16, epochs=200,
                    validation_data=(X_test, y_test))

# After training, load the weights of the best model based on validation accuracy
model.load_weights("best_model_weights.h5")

# User choice for model
user_choice = input("Enter 'svm' to use SVM, 'nn' to use Neural Network: ")

if user_choice.lower() == 'svm':
    # Take input from the user for SVM
    user_input_svm = []

    for column in all_columns:
        value = float(input(f"Enter value for {column}: "))
        user_input_svm.append(value)

    # Convert user input to a NumPy array
    user_input_array_svm = np.array(user_input_svm).reshape(1, -1)

    # Standardize the user input for SVM
    user_input_scaled_svm = svm_pipeline.named_steps['standardscaler'].transform(user_input_array_svm)

    # Predict with the SVM model
    svm_prediction = svm_pipeline.predict(user_input_scaled_svm)[0]

    # Map the prediction to crop name for SVM
    predicted_crop_svm = label_encoder.inverse_transform([svm_prediction])[0]

    # Display the recommended crop for SVM
    print(f"Recommended Crop (SVM): {predicted_crop_svm}")

    # Evaluate accuracy on the test set for SVM
    test_predictions_svm = svm_pipeline.predict(X_test)
    test_accuracy_svm = accuracy_score(y_test, test_predictions_svm)
    print(f"SVM Model Accuracy on Test Set: {test_accuracy_svm * 100}%")

elif user_choice.lower() == 'nn':
    # Take input from the user for Neural Network
    user_input_nn = []

    for column in all_columns:
        value = float(input(f"Enter value for {column}: "))
        user_input_nn.append(value)

    # Convert user input to a NumPy array
    user_input_array_nn = np.array(user_input_nn).reshape(1, -1)

    # Standardize the user input for Neural Network
    user_input_scaled_nn = user_input_array_nn  # No need to scale user input for the neural network

    # Predict with the Neural Network model
    nn_prediction = np.argmax(model.predict(user_input_scaled_nn), axis=1)[0]

    # Map the prediction to crop name for Neural Network
    predicted_crop_nn = label_encoder.inverse_transform([nn_prediction])[0]

    # Display the recommended crop for Neural Network
    print(f"Recommended Crop (Neural Network): {predicted_crop_nn}")

    # Evaluate the model on the test data for Neural Network
    test_loss_nn, test_accuracy_nn = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss_nn}, Test Accuracy: {test_accuracy_nn}')

else:
    print("Invalid choice. Please enter 'svm' or 'nn'.")
