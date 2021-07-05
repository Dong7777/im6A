import pandas as pd
import warnings
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
import scipy.io as sio
import numpy as np

# 读取数据集

# 划分为5折交叉验证数据集

data = sio.loadmat('X.mat')
data = data['X']
a = np.array(data)
df_y = np.array([1] * 4605 + [0] * 4605)
train_sequence = a

train_sequence = train_sequence[:,0:779]
print(train_sequence.shape)
print(type(train_sequence))

train_profile = np.load("dataX.npy")
train_profile=train_profile.reshape(9210,3444)
print(train_profile.shape)
print(type(train_profile))

one_input = np.load("testP.npy")
print(one_input.shape)
print(type(one_input))

df_X=np.hstack([train_sequence,train_profile,one_input])
print(df_X.shape)
df_X=scale(df_X,axis=0)  #将数据转化为标准数据
#构建模型

lr = LogisticRegression(random_state=2018,tol=1e-6)  # 逻辑回归模型

tree = DecisionTreeClassifier(random_state=2018) #决策树模型

svm = SVC(probability=True,random_state=2018,tol=1e-6)  # SVM模型

forest=RandomForestClassifier(n_estimators=100,random_state=2018) #　随机森林

Gbdt=GradientBoostingClassifier(random_state=2018) #CBDT

Xgbc=XGBClassifier(random_state=2018)  #Xgbc

gbm=lgb.LGBMClassifier(random_state=2018)  #lgb



def muti_score(model):
    warnings.filterwarnings('ignore')
    accuracy = cross_val_score(model, df_X, df_y, scoring='accuracy', cv=5)
    precision = cross_val_score(model, df_X, df_y, scoring='precision', cv=5)
    recall = cross_val_score(model, df_X, df_y, scoring='recall', cv=5)
    f1_score = cross_val_score(model, df_X, df_y, scoring='f1', cv=5)
    auc = cross_val_score(model, df_X, df_y, scoring='roc_auc', cv=5)
    print("准确率:",accuracy.mean())
    print("精确率:",precision.mean())
    print("召回率:",recall.mean())
    print("F1_score:",f1_score.mean())
    print("AUC:",auc.mean())



model_name=["lr","tree","svm","forest","Gbdt","Xgbc","gbm"]
for name in model_name:
    model=eval(name)
    print(name)
    muti_score(model)

