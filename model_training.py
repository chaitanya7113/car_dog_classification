import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    RandomFlip, RandomRotation, RandomZoom
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA LOADING
# =========================

train_ds = keras.utils.image_dataset_from_directory(
    directory='data/training_set',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='data/test_set',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(128, 128),
    shuffle=False
)
class_names = validation_ds.class_names
# Normalize images
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
validation_ds = validation_ds.map(lambda x, y: (x/255.0, y))

# =========================
# DATA AUGMENTATION
# =========================

data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.15),
    RandomZoom(0.15)
])

# =========================
# MODEL
# =========================

model = Sequential([
    data_augmentation,

    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# TRAINING
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

class_weight = {
    0: 1.4,  # cats
    1: 1.0   # dogs
}

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=30,
    callbacks=[early_stop],
    class_weight=class_weight
)

# =========================
# PLOTS
model.save("final_cat_dog_model.h5")
print("✅ Model saved as final_cat_dog_model.h5")


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# =========================
# CONFUSION MATRIX (THRESHOLD = 0.4)
# =========================

y_true = []
y_pred_prob = []

for images, labels in validation_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred_prob.extend(preds.flatten())

y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)

threshold = 0.4
y_pred = (y_pred_prob >= threshold).astype(int)



cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)



disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix (threshold = {0.5})')
plt.show()

# =========================
# SAVE MODEL
# =========================

