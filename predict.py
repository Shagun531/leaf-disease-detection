import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to model and class names
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_disease_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

# Load model
model = load_model(MODEL_PATH, compile=False)

# Load class names
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# Prediction function
def predict_leaf(img_path, return_all=False):
    """
    Predicts disease from leaf image.
    
    Args:
        img_path (str): path to image
        return_all (bool): if True, return probabilities for all classes

    Returns:
        If return_all=True: dict of class:probability
        else: (predicted_class, confidence)
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    preds = model.predict(x)[0]
    class_index = np.argmax(preds)
    confidence = preds[class_index]

    if return_all:
        return {cls: float(preds[i]) for i, cls in enumerate(class_names)}
    else:
        return class_names[class_index], confidence
