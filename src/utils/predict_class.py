import os
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

def predict(test_images, model):
    """

    :param path_model:
    :param test_images:
    :return:
    """
    test_images = np.array(test_images)
    n_samples, height, width = test_images.shape
    test_images = test_images.reshape(n_samples, height, width, 1)
    test_images = np.repeat(test_images, repeats=3, axis=-1)

    assert len(test_images) != 0

    # preprocess images
    test_images = preprocess_input(test_images)

    # make predictions
    predictions = model.predict(test_images)

    return predictions