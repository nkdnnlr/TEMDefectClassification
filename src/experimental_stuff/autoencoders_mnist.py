from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard

import numpy as np
import matplotlib.pyplot as plt


def simple_autoencoder():
    # this is the size of our encoded representations
    encoding_dim = (
        32
    )  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation="relu")(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation="sigmoid")(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    autoencoder.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test, x_test),
    )

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def convolutional_autoencoder():
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
    from keras.models import Model
    from keras import backend as K

    input_img = Input(
        shape=(28, 28, 1)
    )  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation="relu")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.reshape(
        x_train, (len(x_train), 28, 28, 1)
    )  # adapt this if using `channels_first` image data format
    x_test = np.reshape(
        x_test, (len(x_test), 28, 28, 1)
    )  # adapt this if using `channels_first` image data format
    print(x_train.shape)
    print(x_test.shape)

    from keras.callbacks import TensorBoard

    autoencoder.fit(
        x_train,
        x_train,
        epochs=1,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[TensorBoard(log_dir="/tmp/autoencoder")],
    )

    decoded_imgs = autoencoder.predict(x_test)

    print(decoded_imgs.shape)

    n = 11
    # plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        # ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        plt.show()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # display reconstruction
        # ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        plt.show()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    # plt.show()


# simple_autoencoder()

convolutional_autoencoder()
