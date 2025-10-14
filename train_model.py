import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt


# Path
DATASET_DIR = "dataset"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "crop_disease_model.h5")
CLASS_PATH = os.path.join(MODELS_DIR, "class_names.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# Dataet load
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Save class names
with open(CLASS_PATH, "w") as f:
    json.dump(class_names, f)
print(f"Class names saved to {CLASS_PATH}")

# Dataset optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Apply preprocess_input 
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def preprocess(images, labels):
    return preprocess_input(images), labels


train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# Model building
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze some layers
for layer in base_model.layers[:-60]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model
EPOCHS = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# result plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.savefig("training_results.png")
plt.show()

print("Training complete and plots saved!")
