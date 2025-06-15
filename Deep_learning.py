import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import sklearn

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# === Load and prepare data ===
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Remove non-feature columns
X_train = train_df.drop(columns=['subject', 'Activity', 'ActivityName'])
X_test = test_df.drop(columns=['subject', 'Activity', 'ActivityName'])

# Encode target labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['ActivityName'])
y_test = label_encoder.transform(test_df['ActivityName'])

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train_cat.shape)

# === Build Deep Neural Network ===
model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train_cat.shape[1], activation='softmax')  # Output layer
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Train Model ===
history = model.fit(
    X_train,
    y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    verbose=1
)

# === Evaluate Model ===
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")

# === Plot Training History ===
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# === Confusion Matrix ===
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_labels)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# === Classification Report ===
print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))
