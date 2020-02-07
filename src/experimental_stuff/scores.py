import numpy as np
import tensorflow as tf
import keras.backend as K


def mse(im1, im2):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
    err /= float(im1.shape[0] * im1.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def ssim(im1, im2):
    """

    :param im1:
    :param im2:
    :return:
    """
    if not tf.is_tensor(im1):
        im1 = tf.constant(im1)
    im2 = tf.cast(im2, im1.dtype)

    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)

    ssim2 = tf.image.ssim(
        im1, im2, max_val=2.0, filter_size=32, filter_sigma=1.5, k1=0.01, k2=0.03
    )

    with tf.Session() as sess:
        ssim2 = ssim2.eval()

    return -1.0 * ssim2 + 1.0


def spectral(y_true, y_pred):
    """
    TODO: UNDER CONSTRUCTION
    :param im1: 
    :param im2: 
    :return: 
    """

    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    # y_true = tf.expand_dims(y_true, 0)
    #
    #     # Filtering
    # y_true = tf.nn.avg_pool2d(y_true, ksize=9, strides=1, padding="SAME")
    # y_pred = tf.nn.avg_pool2d(y_pred, ksize=9, strides=1, padding="SAME")

    # return y_true

    y_true = tf.squeeze(y_true, [-1])
    y_pred = tf.squeeze(y_pred, [-1])

    remember_type = y_pred.dtype

    y_true = tf.dtypes.complex(y_true, 0.0)
    y_pred = tf.dtypes.complex(y_pred, 0.0)
    epsilon = tf.dtypes.complex(K.constant(np.finfo(float).eps), 0.0)

    # Attention: tf.signal.fft2d converts the two inner elements of tensor, where [outer, ..., middle, ..., inner]
    ft_true = tf.signal.fft2d(y_true)
    ft_pred = tf.signal.fft2d(y_pred)
    inner = tf.reduce_sum(tf.math.multiply(ft_pred, tf.math.conj(ft_true)))
    dephase = tf.divide(inner, (tf.norm(inner) + epsilon))

    I_t = tf.signal.ifft2d(tf.multiply(dephase, ft_true))

    I_t = K.cast(I_t, remember_type)
    y_true = K.cast(y_true, remember_type)
    y_pred = K.cast(y_pred, remember_type)

    y_true = tf.expand_dims(y_true, -1)
    y_pred = tf.expand_dims(y_pred, -1)

    I_t = tf.expand_dims(I_t, -1)

    loss = K.mean(K.square(y_pred - I_t), axis=-1)

    with tf.Session() as sess:
        loss = loss.eval()
    # mlflow.log_metric("spectral_loss", loss.eval())
    return loss


def mse_spectral(im1, im2):
    b = 100.0
    b = 1.0
    return 0.0 * mse(im1, im2) + b * spectral(im1, im2)
