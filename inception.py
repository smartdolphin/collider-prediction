import os
import keras
import tensorflow as tf
import numpy as np
import time

from util import save_logs
from util import calculate_metrics
from util import save_test_duration


class INCEPTION:
    def __init__(self, output_directory, data_list, nb_classes=4, verbose=False, build=True, batch_size=256, lr=0.001,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose

        if build == True:
            self.model = self.build_model(data_list, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(os.path.join(self.output_directory, 'model_init.hdf5'))

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, data_list, nb_classes):

        input_list, out_layers = [], []
        for data in data_list:
            input_layer = keras.layers.Input(shape=(data.shape[1:]))
            input_list.append(input_layer)

            x = input_layer
            input_res = input_layer

            for d in range(self.depth):

                x = self._inception_module(x)

                if self.use_residual and d % 3 == 2:
                    x = self._shortcut_layer(input_res, x)
                    input_res = x

            gap_layer = keras.layers.GlobalAveragePooling1D()(x)
            out_layers.append(gap_layer)

        gap_layers = keras.layers.Concatenate()(out_layers)
        fc_layer = keras.layers.Dense(128, activation='relu')(gap_layers)
        fc_layer = keras.layers.Dense(64, activation='relu')(fc_layer)
        fc_layer = keras.layers.Dense(32, activation='relu')(fc_layer)
        fc_layer = keras.layers.Dense(16, activation='relu')(fc_layer)
        output_layer = keras.layers.Dense(nb_classes, activation='linear')(fc_layer)

        model = keras.models.Model(inputs=input_list, outputs=output_layer)

        return model

    def get_model(self):
        return self.model

