from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Reshape, Dense, Convolution1D, Dropout, Input, Flatten, MaxPool1D, add, AveragePooling1D, \
    Bidirectional, GRU, LSTM, Multiply, MaxPooling1D, TimeDistributed, AvgPool1D, Convolution2D, AveragePooling2D, \
    MaxPooling2D
from keras.layers.merge import Concatenate, concatenate
from keras.layers.wrappers import Bidirectional
from six.moves import cPickle as pickle
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, Adamax, Nadam
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.regularizers import l2, l1
from sklearn.metrics import confusion_matrix
from keras import backend as K
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
import os
import sys
import numpy as np
import math
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
import sys

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.utils import to_categorical
import scipy.io as sio
import keras
import os
import pandas as pd
import numpy as np
import pickle
import pdb
import logging, multiprocessing
from collections import namedtuple
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from keras_self_attention import SeqSelfAttention, ScaledDotProductAttention
from scipy import interp
from pandas import read_csv
# import xlrd
from keras import regularizers
from itertools import chain
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from model_Attention import Attention, Capsule
import numpy as np
import scipy.io as sio
import pandas as pd
import keras
import scipy.io as sio
import numpy as np
import pandas as pd
from keras import initializers, layers, regularizers
from keras.layers import Dropout
from keras.models import *
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn import metrics
import os,sys,re
#import keras

from keras import layers, optimizers
from keras.layers import Dropout, Activation,GlobalAveragePooling1D
from keras import callbacks
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from itertools import chain
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings("ignore")
np.random.seed(4)



s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
logs_path = './log_%s' % (s_time)
try:
    os.makedirs(logs_path)
except:
    pass

MODEL_PATH = './'
filepath = os.path.join(MODEL_PATH, 'my_net_model.h5')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

def swish(x, beta=1):  # 定义激活函数
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': swish})

def bn_activation_dropout(input):
    input_bn = BatchNormalization(axis=-1)(input)  # 归一化
    input_at = Activation('relu')(input_bn)  # 激活
    input_dp = Dropout(0.3)(input_at)  # 防止过拟合
    return input_dp


def ConvolutionBlock(input, f, k):  # 滤波器数目，卷积核大小
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
    result = Activation('relu')(result)
    return result


def createModel(filter1,filter2,pool_size,dropout):
    # 一个卷积块加一个嵌入层合并作为残差块的输入 + 平均池化 +GRU+Attention +Flatten +Dense2
    one_input = Input(shape=(41, 4), name='one_input')
    one = Convolution1D(filters=32, kernel_size=3, padding='same')(one_input)
    one = BatchNormalization(axis=-1)(one)
    one = Activation('swish')(one)

    sequence_input = Input(shape=(41, 5), name='sequence_input')
    sequence = Convolution1D(filters=32, kernel_size=3, padding='same')(sequence_input)
    sequence = BatchNormalization(axis=-1)(sequence)
    sequence = Activation('swish')(sequence)

    profile_input = Input(shape=(41, 84), name='profile_input')
    profile = Convolution1D(filters=32, kernel_size=3, padding='same')(profile_input)
    profile = BatchNormalization(axis=-1)(profile)
    profile = Activation('swish')(profile)

    mergeInput = Concatenate(axis=-1)([one, sequence, profile])

    overallResult = MultiScale(mergeInput)
    overallResult = AveragePooling1D(pool_size=pool_size)(overallResult)
    overallResult = Dropout(dropout)(overallResult)
    overallResult = Capsule(num_capsule=14, dim_capsule=41,
                            routings=3, share_weights=True)(overallResult)
    overallResult = Bidirectional(GRU(filter1, return_sequences=True))(overallResult)
    overallResult = Dropout(dropout)(overallResult)
    overallResult = SeqSelfAttention(
        attention_activation='sigmoid',
        name='Attention',
    )(overallResult)
    overallResult = Flatten()(overallResult)
    overallResult = Dense(filter2, activation='relu')(overallResult)
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)

    return Model(inputs=[one_input, sequence_input, profile_input], outputs=[ss_output])


def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.8
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate



tprs = []
mean_fpr = np.linspace(0, 1, 100)

data = sio.loadmat('X.mat')
data = data['X']
a = np.array(data)
ceshitrain_label3 = np.array([1] * 4574 + [0] * 4574).T
ceshitrain_label4 = np.array([0] * 4574 + [1] * 4574).T
train_label = np.vstack([ceshitrain_label3, ceshitrain_label4]).T
train_sequence = a
train_sequence = train_sequence[:,0:205]
train_sequence = train_sequence.reshape(9148, 41, 5)
train_profile = np.load("dataX.npy")
one_input = np.load("testP.npy")
one_input = one_input.reshape(9148, 41,4)
one_input=one_input.astype('float')



kf = KFold(5, True)

for train_index, eval_index in kf.split(train_label):
    train_X1 = train_sequence[train_index]
    train_X2 = train_profile[train_index]
    train_X3 = one_input[train_index]
    train_y = train_label[train_index]
    eval_X1 = train_sequence[eval_index]
    eval_X2 = train_profile[eval_index]
    eval_X3 = one_input[eval_index]
    eval_y = train_label[eval_index]

def rf_cv(filter1,filter2,pool_size,dropout,batchSize):
    model = createModel(filter1=int(filter1),filter2=int(filter2),pool_size=int(pool_size),dropout=dropout)
    model.compile(optimizer='adam',
                  loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,
                                 mode='auto')
    model.fit(
        {'one_input': train_X3, 'sequence_input': train_X1, 'profile_input': train_X2},
        {'ss_output': train_y},
        epochs=100,
        batch_size=int(batchSize),
        callbacks=[EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto'),
                   checkpoint,
                    LearningRateScheduler(step_decay)],
        verbose=0,
        validation_data=({'one_input': eval_X3, 'sequence_input': eval_X1, 'profile_input': eval_X2},
                         {'ss_output': eval_y}))
    predict = model.predict({'one_input': eval_X3, 'sequence_input': eval_X1, 'profile_input': eval_X2})
    AUC = metrics.roc_auc_score(eval_y, predict)
    return AUC
from hyperopt import hp
rf_bo = BayesianOptimization(
    rf_cv,
    {
    'filter1':(16,64),
    'filter2':(16,64),
    'pool_size':(1,5),
    'dropout':(0.1,0.5),
    'batchSize':(16,64),
     }
)
rf_bo.maximize()
print(rf_bo.max['target'])
#查看优化得到的参数
print(rf_bo.max['params'])

























