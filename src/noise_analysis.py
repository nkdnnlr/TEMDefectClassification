import os
from PIL import Image

import numpy as np
from keras.models import load_model

from eigenfilter_segmentation import run
import utils.helpers as helpers




def create_poissonnoisy_images(image_dir, output_dir, a_max=1, a_min=0.001, steps=7):



    for idx, a in enumerate(np.logspace(np.log10(a_max), np.log10(a_min), steps)):

        noise_dir = os.path.join(output_dir, str(np.round(a,4)))
        os.mkdir(noise_dir)

        for name in os.listdir(image_dir):
            file = os.path.join(image_dir, name)
            print(file)
            if not file.endswith('.tif'):
                continue

            image = helpers.get_image(file)
            image_noisy = np.random.poisson(image*a)/a
            im = Image.fromarray(np.uint8(image_noisy))
            im.save(os.path.join(noise_dir, name))

# exit()

def noise_analysis(image_dir, output_dir_parent, model):

    for a in os.listdir(image_dir):

        noise_dir = os.path.join(image_dir, a)
        output_dir = os.path.join(output_dir_parent, a)

        assert os.path.exists(noise_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        run(noise_dir, model, output_dir)


if __name__ == '__main__':

    image_dir = "../data/all_data/defective/images"
    noise_image_dir = "../data/all_data/defective_noise/noise"
    output_dir = "..output/all_data/noise_analysis"

    model_path = "../models/model.h5"
    assert os.path.exists(model_path)
    model = load_model(model_path)

    create_poissonnoisy_images(image_dir, noise_image_dir)
    noise_analysis(noise_image_dir, output_dir, model)
