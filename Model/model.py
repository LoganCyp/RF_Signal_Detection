import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, ReLU, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.utils import to_categorical

X = np.load("cardrf_signals.npy")
y = np.load("cardrf_labels.npy", allow_pickle=True)  

X = np.expand_dims(X, axis=2)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)
class_names = encoder.classes_

X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X, y_cat, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

model = Sequential([
    Conv1D(64, kernel_size=8, padding='same', input_shape=(1024, 1)),
    ReLU(),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=4, padding='same'),
    ReLU(),
    MaxPooling1D(pool_size=2),
    Conv1D(256, kernel_size=2, padding='same'),
    ReLU(),
    GlobalAveragePooling1D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=64
)

plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Predict
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test_enc, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
