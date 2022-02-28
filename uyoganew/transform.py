import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

def transform(image,reconstructed_model,labels):

    img = tf.keras.utils.load_img(
        image, target_size=(224,224), color_mode = "rgb"
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = reconstructed_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    result = labels[np.argmax(score)]
    confidence =  100 * np.max(score)

    return (
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(result,confidence)
    ), result, confidence
