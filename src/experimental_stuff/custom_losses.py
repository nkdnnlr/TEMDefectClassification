import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# from datetime import datetime
# import time
import numpy as np

# import matplotlib.pyplot as plt
# import cv2 as cv2

import mlflow


import tensorflow as tf
from tensorflow.signal import fft2d, ifft2d

# import keras.losses
from keras import backend as K


class CustomLosses:
    def __init__(self, beta=100.0, ksize=None):
        self.beta = beta
        self.ksize = ksize
        self.msemax = 0.2
        pass

    def get_sign(self, mse):
        if self.msemax < mse:
            self.msemax = mse
            return -1.



    def texture_loss(self, y_true, y_pred):
        """
        Texture loss as in s in:
            Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004),
            Image quality assessment: from error visibility to structural similarity,
            IEEE transactions on image processing.
        :param y_true:
        :param y_pred:
        :return:
        """
        if not tf.is_tensor(y_pred):
            y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        im1 = tf.image.convert_image_dtype(y_true, tf.float32)
        im2 = tf.image.convert_image_dtype(y_pred, tf.float32)
        ssim2 = tf.image.ssim(
            im1, im2, max_val=2.0, filter_size=32, filter_sigma=1.5, k1=0.01, k2=0.03
        )
        return K.constant(1.0) + K.constant(-1.0) * ssim2

    def mean_squared_error(self, y_true, y_pred):
        """
        MSE
        :param y_true:
        :param y_pred:
        :return:
        """
        if not K.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)

        loss = K.mean(K.square(y_pred - y_true), axis=-1)
        # mlflow.log_metric("spectral_loss", loss.eval())
        return loss

    def spectral_loss(self, y_true, y_pred):
        """
        Spectral loss as proposed in:
            Gang Liu and Yann Gousseau and Gui{-}Song Xia} (2016)
            Texture Synthesis Through Convolutional Neural Networks and Spectrum Constraints,
            CoRR
        Including filtering
        :param y_true:
        :param y_pred:
        :return:
        """

        if not K.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)

        # y_true = tf.expand_dims(self, y_true, 0)

        if self.ksize is not None:
            # Filtering


            y_true = tf.nn.avg_pool2d(
                y_true, ksize=self.ksize, strides=1, padding="SAME"
            )
            y_pred = tf.nn.avg_pool2d(
                y_pred, ksize=self.ksize, strides=1, padding="SAME"
            )

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

        I_t = ifft2d(tf.multiply(dephase, ft_true))

        I_t = K.cast(I_t, remember_type)
        y_true = K.cast(y_true, remember_type)
        y_pred = K.cast(y_pred, remember_type)

        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)

        I_t = tf.expand_dims(I_t, -1)

        loss = K.mean(K.square(y_pred - I_t), axis=-1)
        # mlflow.log_metric("spectral_loss", loss.eval())
        return loss

    def fourier_loss(self, y_true, y_pred):

        if not K.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)

        remember_type = y_pred.dtype

        y_true = tf.dtypes.complex(y_true, 0.0)
        y_pred = tf.dtypes.complex(y_pred, 0.0)
        epsilon = tf.dtypes.complex(K.constant(np.finfo(float).eps), 0.0)

        ft_true = K.transpose(tf.signal.fft2d(K.transpose(y_true)))
        ft_pred = K.transpose(tf.signal.fft2d(K.transpose(y_pred)))

        norm = tf.norm(ft_true - ft_pred)
        norm = K.cast(norm, remember_type)

        return norm

    def mse_and_fourier(self, y_true, y_pred):
        a = K.constant(1.0)
        beta = K.constant(self.beta)
        return a * self.mean_squared_error(y_true, y_pred) + beta * self.fourier_loss(
            y_true, y_pred
        )

    def mse_and_spectral(self, y_true, y_pred):
        a = K.constant(1.0)
        beta = K.constant(self.beta)

        loss = a * self.mean_squared_error(y_true, y_pred) + beta * (
            self.spectral_loss(y_true, y_pred)
        )
        return loss


if __name__ == "__main__":
    Losses = CustomLosses()


# def spectral_loss(self, y_true, y_pred):
#     if not K.is_tensor(y_pred):
#         y_pred = K.constant(y_pred)
#
#     remember_type = y_pred.dtype
#
#     y_true = K.cast(y_true, y_pred.dtype)
#
#     y_true = tf.dtypes.complex(y_true, 0.)
#     y_pred = tf.dtypes.complex(y_pred, 0.)
#
#     magnitude = tf.dtypes.complex(tf.math.abs(fft2d(y_pred) * tf.math.conj(fft2d(y_true))), 0.)
#
#     nominator = fft2d(y_pred)*tf.math.conj(fft2d(y_true))
#     y_tilde = ifft2d( nominator / magnitude * fft2d(y_true) )
#     y_tilde = K.cast(y_tilde, remember_type)
#     y_pred = K.cast(y_pred, remember_type)
#
#     return K.constant(0.5)*K.sum(K.square(y_pred - y_tilde), axis=-1)
