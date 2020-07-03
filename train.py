import os
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
'''
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten,MaxPooling2D,\
BatchNormalization,Lambda, AveragePooling2D, Dropout, Input, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
'''
import keras
from keras.layers import Dense, Activation, Conv2D, Flatten,MaxPooling2D,\
BatchNormalization,Lambda, AveragePooling2D, Dropout, Input, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

import pandas as pd

from data_loader import get_data
from model import set_model
from util import *


LABEL = ['XY', 'M' , 'V', 'MV']


def train(model, X,Y, epochs=200, batch_size=256, label=None, seq=0, out='./model'):
    MODEL_SAVE_FOLDER_PATH = out
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = os.path.join(MODEL_SAVE_FOLDER_PATH, f'best_{label}_{seq}.hdf5')
    best_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                      patience=10, min_lr=0.0001) 

    history = model.fit(X, Y,
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  validation_split=0.2,
                  verbose = 2,
                  callbacks=[best_save, reduce_lr])

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    '''
    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    plt.title(label)
    plt.savefig(os.path.join(MODEL_SAVE_FOLDER_PATH, 'train.png'))
    #plt.show()    
    '''
    return history 


def run(target, iteration=10, batch_size=256, epochs=500, out='./model'):
    x_train, y_train, x_test = get_data()
    print(f'Kind of Data: {len(x_train)}')
    
    x_pred_list, y_pred_list, m_pred_list, v_pred_list = [], [], [], []
    for i in range(iteration):
        model = set_model(target, x_train)
    
        hist = train(model, x_train, y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     label=LABEL[target],
                     seq=i,
                     out=out)
        best_model = load_best_model(target, label=LABEL[target], seq=i)
        pred_data_test = best_model.predict(x_test)
        if target == 0:
            x_pred_list.append(pred_data_test[:,0])
            y_pred_list.append(pred_data_test[:,1])
        if target == 1 or target == 3:
            m_pred_list.append(pred_data_test[:,2])
        if target == 2 or target == 3:
            v_pred_list.append(pred_data_test[:,3])

    if target == 0:
        x_pred = np.mean(x_pred_list, axis=0)
        y_pred = np.mean(y_pred_list, axis=0)
    if target == 1 or target == 3:
        m_pred = np.mean(m_pred_list, axis=0)
    if target == 2 or target == 3:
        v_pred = np.mean(v_pred_list, axis=0)
    
    # submit
    submit = pd.read_csv('xy_mlp_ensemble10_2nd.csv')
    if target == 0:
        submit.iloc[:1] = x_pred
        submit.iloc[:2] = y_pred
    if target == 1 or target == 3:
        submit.iloc[:3] = m_pred
    if target == 2 or target == 3:
        submit.iloc[:4] = v_pred
    submit.to_csv(os.path.join(out, 'mv_ensemble10.csv'), index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g",  default=0, type=int, help="gpu num(default 0)")
    parser.add_argument("--iter", "-i",  default=10, type=int, help="iteration num(default 10)")
    parser.add_argument("--epochs", "-e",  default=500, type=int, help="epochs(default 500)")
    parser.add_argument("--target", "-t",  default=3, type=int, help="target(default 3)")
    parser.add_argument("--batch_size", "-b",  default=256, type=int, help="batch size(default 256)")
    parser.add_argument("--out", "-o", default="./model", type=str, help="output dir")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f'Use GPU: {args.gpu}')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    run(args.target, args.iter, args.batch_size, args.epochs, args.out)

