import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("TensorFlow version: ", tf.__version__)

# from keras import backend as K
from keras.callbacks import EarlyStopping

# from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras import backend as K


import mlflow

import src.experimental_stuff.scores as custom_scores
import src.experimental_stuff.autoencoder as autoencoder
from src.experimental_stuff import LossHistory, AdditionalValidationGenerators

# import src_.utils.preprocess_crop as preprocess_crop
# import src_.utils.preprocess as preprocess
# import src_.utils.custom_losses as custom_losses


def train_model(train_gen, validation_gen, test_def_gen, test_nondef_gen, output_dir, architecture, epochs=10,
                steps_per_epoch=30, loss=None, es_patience=50, gpu=False):
    """

    :param es_patience:
    :param iterator:
    :return:
    """
    # Define callbacks
    loss_history = LossHistory()
    additional = AdditionalValidationGenerators({"test_defective": test_def_gen,
                                                 "test_nondefective": test_nondef_gen,
                                                 "train_m": train_gen}, verbose=2, steps=32, gpu=gpu)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, min_delta=0.1)

    callbacks = []
    callbacks.append(loss_history)
    callbacks.append(additional)
    callbacks.append(early_stopping)

    # Define optimizer
    optimizer = Adam()

    # Define model architecture
    input_img, decoded = architecture

    # Compile and run model
    model = Model(input_img, decoded)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mse", "mae"])
    model_summary = model.summary()
    plot_model(model=model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    mlflow.log_param("model_summary", model_summary)

    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_gen,
        validation_steps=steps_per_epoch,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
        use_multiprocessing=True
    )
    return model, history


def train_vae(train_gen, validation_gen, test_def_gen, test_nondef_gen, output_dir, loss, depth=2, epochs=10,
              steps_per_epoch=30, gpu=False, expanding=False, filters_init=16):
    # Define callbacks
    loss_history = LossHistory()
    additional = AdditionalValidationGenerators({"test_defective": test_def_gen,
                                                 "test_nondefective": test_nondef_gen,
                                                 "train_m": train_gen}, verbose=2, steps=1, gpu=gpu)
    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, min_delta=0.1)

    callbacks = []
    callbacks.append(loss_history)
    callbacks.append(additional)
    # callbacks.append(early_stopping)


    # instantiate encoder model
    inputs, [z_mean, z_log_var, z], x, shape = autoencoder.vae_encoder(depth=depth, expanding=expanding, filters=filters_init)
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    latent_inputs, outputs = autoencoder.vae_decoder(x, shape, depth=depth, expanding=expanding, filters=filters_init)
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    outputs = decoder(encoder(inputs)[2])

    image_size = 256
    # instantiate VAE model
    # inputs, [z_mean, z_log_var, z], latent_inputs, outputs = autoencoder.vae()
    vae = Model(inputs, outputs, name='vae')

    # Specify loss
    def kl_loss():
        return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)


    def vae_loss(inputss, outputss):
        # reconstruction_loss = mse(K.flatten(inputss), K.flatten(outputss))
        # reconstruction_loss = binary_crossentropy(K.flatten(inputss),
        #                                           K.flatten(outputss))
        reconstruction_loss = tf.reduce_mean(loss(inputss, outputss), axis=[-1, -2])
        reconstruction_loss *= image_size * image_size

        vae_loss = K.mean(reconstruction_loss + kl_loss())
        return vae_loss

    vae.compile(optimizer='adam', loss=vae_loss, metrics=["mse", "mae"])
    vae.summary()
    # plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    # train the autoencoder
    history = vae.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_gen,
        validation_steps=steps_per_epoch,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
        use_multiprocessing=True
    )

    # vae.save_weights('vae_mnist.h5')
    #
    # plot loss history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # testdef_loss = history.history['test_defective_loss']
    # testnondef_loss = history.history['test_nondefective_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    # plt.plot(epochs, testdef_loss, 'r', label='Test-defective loss')
    # plt.plot(epochs, testnondef_loss, 'k', label='Test-nondefective loss')
    # plt.title('Training and validation loss')
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.legend()
    # plt.show()

    # Visualization of latent space
    z_mean, _, _ = encoder.predict_generator(train_gen, steps=16)
    print(z_mean.shape)
    # z_mean, _, _ = encoder.segmentation(x_train, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title('Train Data Latent Space')
    plt.savefig(os.path.join(output_dir, "latentspace_train.png"))
    # plt.show()

    # Visualization of latent space
    z_mean, _, _ = encoder.predict_generator(test_def_gen, steps=16)
    # z_mean, _, _ = encoder.segmentation(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title('Test Data Latent Space')
    plt.savefig(os.path.join(output_dir, "latentspace_train.png"))
    # plt.show()

    return vae, history

    # vae.fit(x_train,
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         validation_data=(x_test, None))
    # vae.save_weights('vae_cnn_mnist.h5')





def test_model(
    model, test_generator, output_dir, score_function="mse", show_plot=False
):
    """

    :param output_dir:
    :param model:
    :param test_generator:
    :return:
    """
    print("Testing...")
    test_images = test_generator.__next__()[0]

    predictions = model.predict(test_images, verbose=1)

    scores = []
    for i in range(len(predictions)):
        print(i)
        if score_function == "mse":
            score = custom_scores.mse(predictions[i], test_images[i])
        elif score_function == "ssim":
            score = custom_scores.ssim(predictions[i], test_images[i])
        elif score_function == "mse_spectral":
            score = custom_scores.mse_spectral(predictions[i], test_images[i])
        else:
            exit()
        scores.append(score)

        plt.subplot(121)
        plt.imshow(test_images[i][:, :, 0], vmin=-1, vmax=1, cmap="gray")
        plt.title("Original")
        plt.subplot(122)
        plt.imshow(predictions[i][:, :, 0], vmin=-1, vmax=1, cmap="gray")
        plt.title("Reconstruction, {}={}".format(score_function, str(round(score, 4))))

        plt.savefig(os.path.join(output_dir, "sample{}.png".format(i)))

        if show_plot:
            plt.show()
        else:
            plt.close()

    mean_score = np.mean(scores)
    print("Mean {}-Score: ".format(score_function), mean_score)
