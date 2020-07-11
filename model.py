import os
import numpy as np
import functools
'''
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Lambda, AveragePooling2D, Dropout, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Add
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
'''
import keras as keras
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D
from keras.layers import BatchNormalization, Lambda, AveragePooling2D, Dropout, Concatenate
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Add
from keras.models import Model
import keras.backend as K
from keras.models import load_model

import pandas as pd
from metrics import my_loss_E1, my_loss_E2, mae_x, mae_y, mae_m, mae_v
import inception
import inceptionv2


def mlp(x, layers):
    for i in range(len(layers)):
        x = Dense(layers[i])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


def xy_model(data):
    input_distance = Input(shape=(data[0].shape[1],))
    input_dist_diff = Input(shape=(data[1].shape[1],))

    distance = mlp(input_distance, layers=[30, 15, 10])
    dist_diff = mlp(input_dist_diff, layers=[60, 30, 10])

    x = Concatenate()([distance, dist_diff])
    x = Dense(64, activation ='relu')(x)
    x = Dense(32, activation = 'relu')(x)
    x = Dense(16, activation ='relu')(x)
    out = Dense(4, activation='linear')(x)
    model = Model(inputs=[input_distance, input_dist_diff],
                  outputs=out)
    return model


def mv_model(data):
    input_org = Input(shape=(data[0].shape[1],5,1))
    input_fft = Input(shape=(data[1].shape[1],5,1))
    input_psd = Input(shape=(data[2].shape[1],5,1))
    input_autocorr = Input(shape=(data[3].shape[1],5,1))
    input_spec = Input(shape=(data[4].shape[1],5,1))

    input_rolling = Input(shape=(data[5].shape[1],))
    input_fftw = Input(shape=(data[6].shape[1],))
    input_psdw = Input(shape=(data[7].shape[1],))
    input_specw = Input(shape=(data[8].shape[1],))
    input_fft_stat = Input(shape=(data[9].shape[1],))
    input_psd_stat = Input(shape=(data[10].shape[1],))
    input_spec_stat = Input(shape=(data[11].shape[1],))
    input_fft_peak = Input(shape=(data[12].shape[1],))
    input_psd_peak = Input(shape=(data[13].shape[1],))
    input_spec_peak = Input(shape=(data[14].shape[1],))
    input_psd_diff = Input(shape=(data[15].shape[1],))
    input_spec_diff = Input(shape=(data[16].shape[1],))
    input_add = Input(shape=(data[17].shape[1],))
    input_v = Input(shape=(data[18].shape[1],))

    # resnet
    org = build_resnet(input_org, input_org.shape, n=5)
    fft = build_resnet(input_fft, input_fft.shape, n=4)
    psd = build_resnet(input_psd, input_psd.shape, n=4)
    ac = build_resnet(input_autocorr, input_autocorr.shape, n=5)
    spec = build_resnet(input_spec, input_spec.shape, n=4)

    roll = mlp(input_rolling, layers=[100, 50, 25])
    fftw = mlp(input_fftw, layers=[100, 50, 25])
    psdw = mlp(input_psdw, layers=[100, 50, 25])
    specw = mlp(input_specw, layers=[100, 50, 25])
    fft_stat = mlp(input_fft_stat, layers=[50, 25, 10])
    psd_stat = mlp(input_psd_stat, layers=[20, 10, 5])
    spec_stat = mlp(input_spec_stat, layers=[20, 10, 5])
    fft_peak = mlp(input_fft_peak, layers=[40, 20, 10])
    psd_peak = mlp(input_psd_peak, layers=[40, 20, 10])
    spec_peak = mlp(input_spec_peak, layers=[40, 20, 10])
    psd_diff = mlp(input_psd_diff, layers=[10, 5])
    spec_diff = mlp(input_spec_diff, layers=[10, 5])
    add = mlp(input_add, layers=[10, 5])
    v = mlp(input_v, layers=[5])

    x = Concatenate()([org, fft, psd, ac, spec,
                       roll, fftw, psdw, specw,
                       fft_stat, psd_stat, spec_stat,
                       fft_peak, psd_peak, spec_peak,
                       psd_diff, spec_diff, add, v
                      ])

#     x = Dropout(0.2)(x)
    x = Dense(128, activation ='relu')(x)
    x = Dense(64, activation ='relu')(x)
    x = Dense(32, activation = 'relu')(x)
    x = Dense(16, activation ='relu')(x)
    out = Dense(4, activation='linear')(x)
    model = Model(inputs=[input_org, input_fft,  input_psd, input_autocorr, input_spec, 
                          input_rolling, input_fftw, input_psdw, input_specw,
                          input_fft_stat, input_psd_stat, input_spec_stat,
                          input_fft_peak, input_psd_peak, input_spec_peak,
                          input_psd_diff, input_spec_diff, input_add, input_v
                         ], 
                  outputs=out)
    return model


def set_model(train_target, data, out='.', name=None):
    if name == 'inception':
        model = inception.INCEPTION(out, data, 4, verbose=True).get_model()
    elif name is not None and (name == 'inceptionv2' or 'inceptionv2' in name):
        model = inceptionv2.INCEPTION(out, data, 4, verbose=True).get_model()
    elif train_target == 0:
        model = xy_model(data)
    else:
        model = mv_model(data)
    optimizer = keras.optimizers.Adam(decay=0.00001)

    if train_target == 0:
        mask = np.array([1,1,0,0])
    elif train_target == 1:
        mask = np.array([0,0,1,0])
    elif train_target == 2:
        mask = np.array([0,0,0,1])
    else:
        mask = np.array([0,0,1,1])
       
    if train_target==0:
        loss_func = functools.partial(my_loss_E1, mask=mask)
        loss_func.__name__ = 'my_loss_E1'
        model.compile(loss=loss_func,
                      optimizer=optimizer,
                      metrics=[mae_x, mae_y]
                     )
    elif train_target==1:
        loss_func = functools.partial(my_loss_E2, mask=mask)
        loss_func.__name__ = 'my_loss_E2'
        model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=[mae_m]
                 )    
    elif train_target==2:
        loss_func = functools.partial(my_loss_E2, mask=mask)
        loss_func.__name__ = 'my_loss_E2'
        model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=[mae_v]
                 )    
    else:
        loss_func = functools.partial(my_loss_E2, mask=mask)
        loss_func.__name__ = 'my_loss_E2'
        model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=[mae_m, mae_v]
                 )       
    model.summary()
    return model


def build_resnet(input, input_shape=(375,5,1), n_feature_maps=8, n=5):
    conv_x = BatchNormalization()(input)
    
    for i in range(n):
        d = 2**i
        if i != 0:
            x1 = y
        else:
            x1 = conv_x
        #print ('build conv_x')
        conv_x = Conv2D(n_feature_maps*d, (8, 1), padding='same')(x1)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('elu')(conv_x)

        #print ('build conv_y')
        conv_y = Conv2D(n_feature_maps*d, (5, 1), padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('elu')(conv_y)

        #print ('build conv_z')
        conv_z = Conv2D(n_feature_maps*d, (3, 1), padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)
        conv_z = MaxPooling2D(pool_size=(2,1))(conv_z)

        is_expand_channels = not (input_shape[-1] == n_feature_maps*d)
        if is_expand_channels:
            shortcut_y = Conv2D(n_feature_maps*d, (1, 1),padding='same')(x1)
            shortcut_y = BatchNormalization()(shortcut_y)
            shortcut_y = MaxPooling2D(pool_size=(2,1))(shortcut_y)
        else:
            shortcut_y = BatchNormalization()(x1)
            shortcut_y = MaxPooling2D(pool_size=(2,1))(shortcut_y)
        #print ('Merging skip connection')
        y = Add()([shortcut_y, conv_z])
        y = Activation('elu')(y)
     
    out = GlobalMaxPooling2D()(y)
#     out = GlobalAveragePooling2D()(y)
#     out = Flatten()(full)
#    print ('        -- model was built.')
    return out


def load_best_model(train_target, is_train=False, label=None, seq=0, out='.'):
    model_path = os.path.join(out, f'best_{label}_{seq}.hdf5')
    print(f'Loading {model_path}..')

    if train_target == 0:
        model = load_model(model_path, custom_objects={'my_loss_E1': my_loss_E1,
                                                       'mae_x': mae_x, 'mae_y': mae_y})
    elif train_target == 1:
        mask = np.array([0,0,1,0])
        loss_func = functools.partial(my_loss_E2, mask=mask)
        loss_func.__name__ = 'my_loss_E2'
        model = load_model(model_path, custom_objects={'my_loss_E2': loss_func,
                                                       'mae_m': mae_m})
    elif train_target == 2:
        mask = np.array([0,0,0,1])
        loss_func = functools.partial(my_loss_E2, mask=mask)
        loss_func.__name__ = 'my_loss_E2'
        model = load_model(model_path, custom_objects={'my_loss_E2': loss_func,
                                                      'mae_v': mae_v})
    else:
        model = load_model(model_path, custom_objects={'my_loss_E2': my_loss_E2,
                                                       'mae_m': mae_m, 'mae_v': mae_v})
    return model


if __name__ == '__main__':
    from data_loader import get_data
    x_train, _, _ = get_data()
    set_model(3, x_train)

