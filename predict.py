"""
predict.py
----------
Load the trained classifier and predict "stable" vs "risk" for a new image.
This is what your Flask backend will call whenever a new camera image arrives.
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/slope_classifier.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["risk", "stable"]  # alphabetical order, matches image_dataset_from_directory

_model_cache = None  # loaded once, reused across calls (Flask calls this per upload)


def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = tf.keras.models.load_model(MODEL_PATH)
    return _model_cache


def predict_image(image_path):
    model = _get_model()

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_input(img_array)  # must match train_classifier.py's
                                               # preprocessing exactly -- MobileNetV2
                                               # expects [-1,1], not raw [0,1] rescale
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {"class": predicted_class, "confidence": round(confidence, 4)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    result = predict_image(sys.argv[1])
    print(result)
