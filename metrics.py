import numpy as np
import keras.backend as K
from keras.layers import Lambda


np.random.seed(0)

weight1 = np.array([1,1,0,0])
weight2 = np.array([0,0,1,1])
weightm = np.array([1,0])
weightv = np.array([0,1])

weight_m = np.array([0, 0, 1, 0])
weight_v = np.array([0, 0, 0, 1])

def my_loss(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult))


def my_loss_E1(y_true, y_pred, mask=weight1):
    return K.mean(K.square(y_true-y_pred)*mask)/2e+04


def my_loss_E2(y_true, y_pred, mask=weight2):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+K.epsilon())])
    return K.mean(K.square(divResult)*mask)


def mspe(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+K.epsilon())])
    return K.mean(K.square(divResult))


def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


def mae_m(y_true, y_pred): 
    return K.mean(K.abs(y_pred - y_true)*weight_m)


def mae_v(y_true, y_pred):    
    return K.mean(K.abs(y_pred - y_true)*weight_v)


def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)


def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


def E2M(y_true, y_pred):    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06))*weightm, axis = 1))


def E2V(y_true, y_pred):    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06))*weightv, axis = 1))

