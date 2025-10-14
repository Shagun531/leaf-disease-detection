import tensorflow as tf
import json, os

data_dir = "dataset"
os.makedirs("models", exist_ok=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes found:", class_names)

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

print("class_names.json created successfully inside models/")
