import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences

from src.data_preparation import process_metadata
from src.feature_extraction import features_extractor
from src.model import build_model, train_model
from src.utils import plot_training_history, plot_confusion_matrix, plot_class_accuracy

# Load and process data
metadata = pd.read_csv("information.csv")
metadata = metadata.sample(len(metadata))

extracted_features = process_metadata(metadata)
extracted_features_df = pd.DataFrame(extracted_features, columns=['Features', 'Class'])
extracted_features_df = extracted_features_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare features and labels
x = np.array(extracted_features_df['Features'].tolist())
y = label_encoder.fit_transform(extracted_features_df['Class'].tolist())

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Scale the features
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Reshape for CNN
x_train_scaled = np.expand_dims(x_train_scaled, axis=-1)
x_test_scaled = np.expand_dims(x_test_scaled, axis=-1)

# Build and train the model
model = build_model(input_shape=(x_train_scaled.shape[1], x_train_scaled.shape[2]), num_classes=len(np.unique(y_train)))
history = train_model(model, x_train_scaled, y_train, x_test_scaled, y_test)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Plot results
plot_training_history(history)
plot_confusion_matrix(model, x_test_scaled, y_test, label_encoder.classes_)
plot_class_accuracy(y_test, model.predict(x_test_scaled), label_encoder.classes_)

# Save the model
model.save('bird_classification_model.h5')
