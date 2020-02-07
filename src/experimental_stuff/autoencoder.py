import tensorflow as tf
import keras.layers
from keras.layers import (
    Input,
    Dense,
    Flatten,
    Reshape,
    Conv2D,
    Deconvolution2D,
    MaxPooling2D,
    Lambda,
    UpSampling2D,
    MaxPooling2D,
    Conv2DTranspose,
    BatchNormalization,
)
from keras.models import Model
from keras import regularizers
from keras import backend as K


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))  # by default, random_normal has mean=0 and std=1.0
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def ae_expanding(depth, filters_init, dense=None):
    # Network settings
    kernel_size = 3
    strides = 2
    filter = filters_init
    regularizer = None #regularizers.l1()

    # Encoder
    input_img = Input(shape=(256, 256, 1))  # adapt this if using `channels_first` image data format
    x = input_img
    for i in range(depth):
        x = Conv2D(filter, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(x)
        filter *= 2
    encoded = x

    # Dense Layers
    if dense is not None:
        b, h, w, f = x.shape
        # h = int(h)
        # w = int(w)
        # f = int(f)
        # print(b, h, w, f)
        # exit()
        flat = Flatten()(x)
        x = flat
        for i in range(dense):
            x = Dense((h*w*f), activation="relu")(x)
        x = Reshape((h, w, f))(x)

    # Decoder
    for i in range(depth):
        filter //= 2
        x = Conv2DTranspose(filter, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(x)
    decoded = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='tanh',
                              padding='same',
                              name='decoder_output')(x)
    return input_img, decoded


def ae_contracting(depth, filters_init, dense=None):
    kernel_size = 3
    strides = 2
    filter = filters_init

    input_img = Input(shape=(256, 256, 1))  # adapt this if using `channels_first` image data format


    #
    # if fourier:
    #     # Fourier transformation of image
    #     f = np.fft.fft2(img)
    #     fshift = np.fft.fftshift(f)
    #     magnitude_spectrum = 20 * np.log(np.abs(fshift))
    #
    #     input_img = tf.dtypes.complex(input_img, 0.0)
    #     input_img = tf.signal.fft2d(input_img)
    #     input_img = tf.signal.



    # Encoder
    x = input_img
    for i in range(depth):
        x = Conv2D(filter, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(x)
        filter //= 2
    encoded = x

    # Dense Layers
    if dense is not None:
        b, h, w, f = x.shape
        h = int(h)
        w = int(w)
        f = int(f)
        print(b, h, w, f)
        # exit()
        flat = Flatten()(x)
        x = flat
        for i in range(dense):
            x = Dense((h*w*f), activation="relu")(x)
        x = Reshape((h, w, f))(x)

    # Decoder
    for i in range(depth):
        filter *= 2
        x = Conv2DTranspose(filter, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(x)
    decoded = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='tanh',
                              padding='same',
                              name='decoder_output')(x)
    return input_img, decoded



def fullyconnected():
    input_img = Input(
        shape=(256, 256, 1)
    )  # adapt this if using `channels_first` image data format
    x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(input_img)
    # x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    # x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    # x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    decoded = Conv2D(1, (3, 3), activation="tanh", padding="same")(x)

    return input_img, decoded

def fullyconnected_stochastic():
    input_img = Input(
        shape=(256, 256, 1)
    )  # adapt this if using `channels_first` image data format
    x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(input_img)
    # x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    # x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    # x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    decoded = Conv2D(1, (3, 3), activation="tanh", padding="same")(x)

    return input_img, decoded


def vae_encoder(depth=2, latent_dim=2, filters=16, expanding=True):
    # network parameters
    input_shape = (256, 256, 1)
    # batch_size = 8 #128
    kernel_size = 3
    filters = filters
    latent_dim = latent_dim
    # epochs = 30

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(depth):
        if expanding:
            filters *= 2
        else:
            filters //= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    return inputs, [z_mean, z_log_var, z], x, shape


def vae_decoder(x, shape, depth=2, latent_dim=2, filters=16, expanding=True):
    # build decoder model

    # network parameters
    input_shape = (256, 256, 1)
    # batch_size = 8 #128
    kernel_size = 3
    if expanding:
        filters *= 2**depth
    else:
        filters //= 2**depth

    latent_dim = latent_dim
    # epochs = 30

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(depth):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        if expanding:
            filters //= 2
        else:
            filters *= 2

    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='tanh',
                              padding='same',
                              name='decoder_output')(x)

    return latent_inputs, outputs


def fullyconnected_fourier():
    input_img = Input(
        shape=(256, 256, 1)
    )  # adapt this if using `channels_first` image data format
    x = Lambda(
        lambda v: tf.cast(
            tf.transpose(
                tf.spectral.fft2d(tf.transpose(tf.cast(v, dtype=tf.complex64)))
            ),
            tf.float32,
        )
    )(input_img)
    # x = Flatten()(x)
    # x = Dense((256*256), activation="relu")(x)
    # x = Reshape((256, 256, 1))(x)

    # x = Reshape((256, 256, 1))(x)
    # x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    # x = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = Conv2D(32, (7, 7), strides=1, activation="relu", padding="same")(x)
    x = Lambda(
        lambda v: tf.cast(
            tf.transpose(
                tf.spectral.ifft2d(tf.transpose(tf.cast(v, dtype=tf.complex64)))
            ),
            tf.float32,
        )
    )(x)
    decoded = Conv2D(1, (3, 3), activation="tanh", padding="same")(x)

    return input_img, decoded


if __name__ == "__main__":
    pass
    # input_img, decoded, encoded = expanding_dense_depth4()
    # print(input_img.shape)
    # print(encoded.shape)
    # print(decoded.shape)
