
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
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
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

import numpy as np
from keras.utils import np_utils
from keras.models import load_model

np.random.seed(4)

def swish(x, beta=1):
    return (x * sigmoid(beta * x))


get_custom_objects().update({'swish': swish})

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

    sequence_input = Input(shape=(41, 15), name='sequence_input')
    sequence = Convolution1D(filters=32, kernel_size=3, padding='same')(sequence_input)
    sequence = BatchNormalization(axis=-1)(sequence)
    sequence = Activation('swish')(sequence)

    profile_input = Input(shape=(41, 84), name='profile_input')
    profile = Convolution1D(filters=32, kernel_size=3, padding='same')(profile_input)
    profile = BatchNormalization(axis=-1)(profile)
    profile = Activation('swish')(profile)

    mergeInput = Concatenate(axis=-1)([one, sequence, profile])

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

    return Model(inputs=[one_input, sequence_input, profile_input], outputs=[ss_output])

ceshidata = sio.loadmat('X.mat')
ceshidata = ceshidata['X']
ceshitrain_label = np.array([1] * 2352 + [0] * 2352)
ceshidata = ceshidata[:,0:615]
ceshidata = ceshidata.reshape(4704, 41, 15)
ceshitrain_profile = np.load("dataX.npy")
one_testinput = np.load("testP.npy")
one_testinput = one_testinput.reshape(4704, 41,4)
one_testinput =one_testinput .astype('float')

model = createModel()
model.summary()
model.load_weights('model2.h5')
model.compile(optimizer='adam',
              loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])
layer1_model = Model(inputs = model.input, outputs = model.get_layer('concatenate_1').output)
result = layer1_model.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})
np.save('concatenate_1.npy',result)

layer1_model = Model(inputs = model.input, outputs = model.get_layer('concatenate_2').output)
result = layer1_model.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})
np.save('concatenate_2.npy',result)

layer1_model = Model(inputs = model.input, outputs = model.get_layer('capsule_1').output)
result = layer1_model.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})
np.save('capsule_1.npy',result)

layer1_model = Model(inputs = model.input, outputs = model.get_layer('Attention').output)
result = layer1_model.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})
np.save('Attention.npy',result)

layer1_model = Model(inputs = model.input, outputs = model.get_layer('dense_1').output)
result = layer1_model.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})
np.save('dense_1',result)
# model2 = createModel()
# model2.load_weights('model1.h5')
# model2.compile(optimizer='adam',
#               loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])
#
# y_predict2 =model2.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})
#
# y_predce2 = np.argmin(y_predict2, axis=1)
#
# ytruece = ceshitrain_label
#
# (Sn2, Sp2, Acc2, MCC2)=Twoclassfy_evalu(ytruece, y_predce2)
#
#
# model3 = createModel()
# model3.load_weights('model2.h5')
# model3.compile(optimizer='adam',
#               loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])
#
# y_predict3 =model3.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})
#
# y_predce3= np.argmin(y_predict3, axis=1)
#
# ytruece = ceshitrain_label
#
# (Sn3, Sp3, Acc3, MCC3)=Twoclassfy_evalu(ytruece, y_predce3)
#
#
# model4 = createModel()
# model4.load_weights('model3.h5')
# model4.compile(optimizer='adam',
#               loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])
#
# y_predict4 =model4.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})
#
# y_predce4 = np.argmin(y_predict4, axis=1)
#
# ytruece = ceshitrain_label
#
# (Sn4, Sp4, Acc4, MCC4)=Twoclassfy_evalu(ytruece, y_predce4)
#
#
# model5 = createModel()
# model5.load_weights('model4.h5')
# model5.compile(optimizer='adam',
#               loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])
#
# y_predict5 =model5.predict({'one_input':one_testinput,'sequence_input': ceshidata, 'profile_input': ceshitrain_profile})
#
# y_predce5 = np.argmin(y_predict5, axis=1)
#
# ytruece = ceshitrain_label
#
# (Sn5, Sp5, Acc5, MCC5)=Twoclassfy_evalu(ytruece, y_predce5)
# m=[y_predce1,y_predce2,y_predce3,y_predce4,y_predce5]
# np.save("m.npy", m)
# m = np.load("m.npy")
# a = m.sum(axis=0)
# for i in range(len(a)):
#     if a[i] > 2:
#         a[i] = 1
#     else:
#         a[i] = 0
# ytruece = ceshitrain_label
# (Sn, Sp, Acc, MCC)=Twoclassfy_evalu(ytruece, a)
# print(Sn, Sp,MCC, Acc)

# print(Sn2, Sp2,MCC2, Acc2)
# print(Sn3, Sp3,MCC3, Acc3)
# print(Sn4, Sp4,MCC4, Acc4)
# print(Sn5, Sp5,MCC5, Acc5)

