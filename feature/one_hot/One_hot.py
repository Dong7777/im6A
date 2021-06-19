import pandas as pd
import numpy as np
from numpy import argmax
from numpy import array
import keras
from keras.utils import to_categorical
def read_fasta_file():

    fh = open('h_b_all.fa', 'r')

    seq = []
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            seq.append(line.replace('\n', '').replace('\r', ''))

    fh.close()
    matrix_data = np.array([list(e) for e in seq]) #列出每个序列中的核苷酸
    print(matrix_data)
    print(len(matrix_data))
    return matrix_data

def extract_line(data_line):
    A=[0,0,0,1]
    U=[0,0,1,0]
    C=[0,1,0,0]
    G=[1,0,0,0]
    
    feature_representation={"A":A,"C":C,"G":G,"U":U }
    one_line_feature=[]
    for index,data in enumerate(data_line):
#        print(index,data)
        if data in feature_representation.keys():   
#        print (index,data) #每一个序列的每个核苷酸代表一个索引，为 0-40（0 A……40 T）
#        print(feature_representation[data])#每个核苷酸的编码#[1, 0, 0, 0]#[0, 1, 0, 0]
            one_line_feature.extend(feature_representation[data])
    return one_line_feature   

def feature_extraction(matrix_data):    
    final_feature_matrix=[extract_line(e) for e in matrix_data]
    return final_feature_matrix

matrix_data = read_fasta_file()
#print(matrix_data)
final_feature_matrix = feature_extraction(matrix_data)
#print(final_feature_matrix)
print(np.array(final_feature_matrix).shape)    
#pd.DataFrame(final_feature_matrix).to_csv('G:/siRNA_Characteristics/one_hot/save_data/trainP.csv',header=None,index=False)
#pd.DataFrame(final_feature_matrix).to_csv('G:/siRNA_Characteristics/one_hot/save_data/trainN.csv',header=None,index=False)
pd.DataFrame(final_feature_matrix).to_csv('testP.csv',header=None,index=False)
#pd.DataFrame(final_feature_matrix).to_csv('G:/siRNA_Characteristics/one_hot/save_data/testN.csv',header=None,index=False)
import csv
#csv_file=csv.reader(open('G:/siRNA_Characteristics/one_hot/save_data/trainP.csv','r')) 
#csv_file=csv.reader(open('G:/siRNA_Characteristics/one_hot/save_data/trainN.csv','r')) 
csv_file=csv.reader(open('testP.csv','r'))
#csv_file=csv.reader(open('G:/siRNA_Characteristics/one_hot/save_data/testN.csv','r')) 
print (csv_file)

final_feature_matrix1 = np.array(final_feature_matrix)
#np.save("G:/siRNA_Characteristics/one_hot/save_data/trainP.npy",final_feature_matrix1)
#np.save("G:/siRNA_Characteristics/one_hot/save_data/trainN.npy",final_feature_matrix1)
np.save("testP.npy",final_feature_matrix1)
#np.save("G:/siRNA_Characteristics/one_hot/save_data/testN.npy",final_feature_matrix1)







