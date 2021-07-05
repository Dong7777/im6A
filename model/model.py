from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Reshape, Dense, Convolution1D, Dropout, Input, Flatten, MaxPool1D, add, AveragePooling1D, \
    Bidirectional, GRU, LSTM, Multiply, MaxPooling1D, TimeDistributed, AvgPool1D,Convolution2D,AveragePooling2D,MaxPooling2D
from keras.layers.merge import Concatenate, concatenate
from keras.layers.wrappers import Bidirectional

from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, Adamax, Nadam
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.regularizers import l2, l1
from sklearn.metrics import confusion_matrix

from keras.backend import sigmoid
from keras import metrics
from keras.constraints import max_norm
import logging
import os
import sys
import numpy as np
import time
import argparse
import math
import logging
import tensorflow as tf
import collections
from itertools import cycle
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from keras.engine.topology import Layer
from keras.utils.generic_utils import get_custom_objects
from keras.layers.core import Lambda
from keras.layers import dot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import scipy.io as sio
import keras
import pandas as pd
import pickle
import pdb
import logging, multiprocessing
from sklearn.model_selection import train_test_split
from keras_self_attention import SeqSelfAttention, ScaledDotProductAttention
from scipy import interp
from pandas import read_csv
from keras import regularizers
from itertools import chain
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from model_Attention import Attention, Capsule


np.random.seed(4)
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': swish})

def Twoclassfy_evalu(y_test, y_predict):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    for i in range(len(y_test)):
        if y_predict[i] > 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 0:
            TN += 1
    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TN + FN) * (FP + TN) * (TP + FN) * (TP + FP))
    Acc = (TP + TN) / (TP + FP + TN + FN)

    return Sn, Sp, Acc, MCC


def bn_activation_dropout(input):
    input_bn = BatchNormalization(axis=-1)(input)
    input_at = Activation('swish')(input_bn)
    input_dp = Dropout(0.3)(input_at)
    return input_dp


def ConvolutionBlock(input, f, k):
    A1 = Convolution1D(filters=f, kernel_size=k, padding='same')(input)
    A1 = bn_activation_dropout(A1)
    return A1


def MultiScale(input):
    A = ConvolutionBlock(input, 32, 1)
    C = ConvolutionBlock(input, 32, 1)
    D = ConvolutionBlock(input, 32, 1)
    D = ConvolutionBlock(D, 32, 5)
    merge = Concatenate(axis=-1)([A, C, D])
    shortcut_y = Convolution1D(filters=96, kernel_size=1, padding='same')(input)
    shortcut_y = BatchNormalization()(shortcut_y)
    result = add([shortcut_y, merge])
    result = Activation('swish')(result)
    return result


def createModel():
    one_input = Input(shape=(41, 4), name='one_input')
    one = Convolution1D(filters=32, kernel_size=3, padding='same')(one_input)
    one = BatchNormalization(axis=-1)(one)
    one = Activation('swish')(one)
    
    sequence_input = Input(shape=(41, 19), name='sequence_input')
    sequence = Convolution1D(filters=32, kernel_size=3, padding='same')(sequence_input)
    sequence = BatchNormalization(axis=-1)(sequence)
    sequence = Activation('swish')(sequence)
    
    profile_input = Input(shape=(41, 84), name='profile_input')
    profile = Convolution1D(filters=32, kernel_size=3, padding='same')(profile_input)
    profile = BatchNormalization(axis=-1)(profile)
    profile = Activation('swish')(profile)
    
    mergeInput = Concatenate(axis=-1)([one ,sequence, profile])
    
    overallResult = MultiScale(mergeInput)
    overallResult = AveragePooling1D(pool_size=3)(overallResult)
    overallResult = Dropout(0.3)(overallResult)
    overallResult = Capsule(num_capsule=14, dim_capsule=41,
                            routings=3, share_weights=True)(overallResult)
    overallResult = Bidirectional(GRU(32, return_sequences=True))(overallResult)
    overallResult = Dropout(0.3)(overallResult)
    overallResult = SeqSelfAttention(
        attention_activation='sigmoid',
        name='Attention',
    )(overallResult)
    overallResult = Flatten()(overallResult)
    overallResult = Dense(16, activation='relu')(overallResult)
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)

    return Model(inputs=[one_input,sequence_input, profile_input], outputs=[ss_output])


def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.8
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

MODEL_PATH = './'
filepath = os.path.join(MODEL_PATH, 'my_net_model.h5')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

tprs = []
mean_fpr = np.linspace(0, 1, 100)
data = sio.loadmat('X.mat')
data = data['X']
a = np.array(data)
ceshitrain_label3 = np.array([1] * 4605 + [0] * 4605).T
ceshitrain_label4 = np.array([0] * 4605 + [1] * 4605).T
train_label = np.vstack([ceshitrain_label3, ceshitrain_label4]).T
train_sequence = a
train_sequence = train_sequence[:,0:779]
train_sequence = train_sequence.reshape(9210, 41, 19)
train_profile = np.load("dataX.npy")
one_input = np.load("testP.npy")
one_input = one_input.reshape(9210, 41,4)
one_input=one_input.astype('float')
batchSize = 32
maxEpochs = 100
kf = KFold(5, True)
ycvSn=[]
ycvSp=[]
ycvMCC=[]
ycvAcc=[]
cvSn=[]
cvSp=[]
cvMCC=[]
cvAcc=[]
i = 0
m = []
ceshidata = sio.loadmat('ceshiX.mat')
ceshidata = ceshidata['ceshiX']
ceshitrain_label = np.array([1] * 4604 + [0] * 4604)
ceshidata = ceshidata[:,0:779]
ceshidata = ceshidata.reshape(9208, 41, 19)
ceshitrain_profile = np.load("indataX.npy")
one_testinput = np.load("test.npy")
one_testinput = one_testinput.reshape(9208, 41,4)
one_testinput =one_testinput .astype('float')

for train_index, eval_index in kf.split(train_label):
    s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logs_path = './log_%s' % (s_time)
    try:
        os.makedirs(logs_path)
    except:
        pass

    train_X1 = train_sequence[train_index]
    train_X2 = train_profile[train_index]
    train_X3 = one_input [train_index]
    train_y = train_label[train_index]
    eval_X1 = train_sequence[eval_index]
    eval_X2 = train_profile[eval_index]
    eval_X3 = one_input [eval_index]
    eval_y = train_label[eval_index]

    i_str = str(i)
    a = train_X1.reshape(train_X1.shape[0], train_X1.shape[1] * train_X1.shape[2])
    test1 = pd.DataFrame(data=a)
    f = open(i_str + 'train_X1.csv', 'w')
    test1.to_csv(i_str + 'train_X1.csv', encoding='gbk')
    f.close()

    b = train_X2.reshape(train_X2.shape[0], train_X2.shape[1] * train_X2.shape[2])
    test2 = pd.DataFrame(data=b)
    f = open(i_str + 'train_X2.csv', 'w')
    test2.to_csv(i_str + 'train_X2.csv', encoding='gbk')
    f.close()

    test5 = pd.DataFrame(data=train_y)
    f = open(i_str + 'train_y', 'w')
    test5.to_csv(i_str + 'train_y.csv', encoding='gbk')
    f.close()

    c = eval_X1.reshape(eval_X1.shape[0], eval_X1.shape[1] * eval_X1.shape[2])
    test3 = pd.DataFrame(data=c)
    f = open(i_str + 'eval_X1', 'w')
    test3.to_csv(i_str + 'eval_X1.csv', encoding='gbk')
    f.close()

    d = eval_X2.reshape(eval_X2.shape[0], eval_X2.shape[1] * eval_X2.shape[2])
    test4 = pd.DataFrame(data=d)
    f = open(i_str + 'eval_X2.csv', 'w')
    test4.to_csv(i_str + 'eval_X2.csv', encoding='gbk')
    f.close()

    test6 = pd.DataFrame(data=eval_y)
    f = open(i_str + 'eval_y', 'w')
    test6.to_csv(i_str + 'eval_y.csv', encoding='gbk')
    f.close()


    n = eval_X3.reshape(eval_X3.shape[0], eval_X3.shape[1] * eval_X3.shape[2])
    test7 = pd.DataFrame(data=n)
    f = open(i_str + 'eval_X3.csv', 'w')
    test7.to_csv(i_str + 'eval_X3.csv', encoding='gbk')
    f.close()

    p = train_X3.reshape(train_X3.shape[0], train_X3.shape[1] * train_X3.shape[2])
    test8 = pd.DataFrame(data=p)
    f = open(i_str + 'train_X3.csv', 'w')
    test8.to_csv(i_str + 'train_X3.csv', encoding='gbk')
    f.close()
     

    model = createModel()
    model.count_params()
    model.summary()
    model.compile(optimizer='adam',
                  loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,
                                 mode='auto')
    history = model.fit(
        {'one_input':train_X3 ,'sequence_input': train_X1, 'profile_input': train_X2},
        {'ss_output': train_y},
        epochs=maxEpochs,
        batch_size=batchSize,
        callbacks=[EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto', restore_best_weights=True),
                   checkpoint,
                   tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True,
                                                  write_images=True), LearningRateScheduler(step_decay)],
        verbose=2,
        validation_data=({'one_input':eval_X3,'sequence_input': eval_X1, 'profile_input': eval_X2},
                         {'ss_output': eval_y}),
        shuffle=True)

    model.save_weights('model' + str(i) + '.h5')

    i = i + 1

    score = model.evaluate({'one_input':eval_X3,'sequence_input': eval_X1, 'profile_input': eval_X2}, eval_y)

    y_pred1 = model.predict({'one_input':eval_X3,'sequence_input': eval_X1, 'profile_input': eval_X2})

    y_pred = np.argmin(y_pred1, axis=1)

    ytrue = np.argmin(eval_y, axis=1)

    (Sn1, Sp1, Acc1, MCC1)=Twoclassfy_evalu(ytrue, y_pred)

    ycvSn.append(Sn1 * 100)
    ycvSp.append(Sp1 * 100)
    ycvMCC.append(MCC1 * 100)
    ycvAcc.append(Acc1 * 100)

    y_predict = model.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})

    y_predce = np.argmin(y_predict, axis=1)

    ytruece = ceshitrain_label

    (Sn2, Sp2, Acc2, MCC2)=Twoclassfy_evalu(ytruece, y_predce)

    cvSn.append(Sn2 * 100)
    cvSp.append(Sp2 * 100)
    cvMCC.append(MCC2 * 100)
    cvAcc.append(Acc2 * 100)

    m.append(y_predce)

np.save("m.npy", m)
m = np.load("m.npy")
a = m.sum(axis=0)
for i in range(len(a)):
    if a[i] > 2:
        a[i] = 1
    else:
        a[i] = 0
ytruece = ceshitrain_label
(Sn, Sp, Acc, MCC)=Twoclassfy_evalu(ytruece, a)
np.save('m.npy', m)


print('yceSn',ycvSn)
print('yceSp',ycvSp)
print('yceMCC',ycvMCC)
print('yceAcc',ycvAcc)
ymeanSn = np.mean(ycvSn)
ymeanSp = np.mean(ycvSp)
ymeanMCC = np.mean(ycvMCC)
ymeanAcc = np.mean(ycvAcc)

print('yceSn',ymeanSn)
print('yceSp',ymeanSp)
print('yceMCC',ymeanMCC)
print('yceAcc',ymeanAcc)

print('ceSn',cvSn)
print('ceSp',cvSp)
print('ceMCC',cvMCC)
print('ceAcc',cvAcc)

meanSn = np.mean(cvSn)
meanSp = np.mean(cvSp)
meanMCC = np.mean(cvMCC)
meanAcc = np.mean(cvAcc)
print('ceSn',meanSn)
print('ceSp',meanSp)
print('ceMCC',meanMCC)
print('ceAcc',meanAcc)

print('Sn', Sn)
print('Sp', Sp)
print('MCC', MCC)
print('ACC', Acc)





