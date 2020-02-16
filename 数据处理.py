import urllib.request
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
path='E:/data/titanic3.xls'
all_df=pd.read_excel(path)
#print(all_df[:2])#数据展示
#ticket（船票号码）和cabin（舱位号码）与预测结果无关，将其忽略。
cols=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
all_df=all_df[cols]
#print(all_df[:2])#数据展示
'''
name 姓名字段在训练时不需要，必须先删除，但在预测阶段会使用
age 有些项的age字段是null，必须将null改为平均值
fare 同age
sex 性别字段是文字，需转换为0和1
embarked 登船港口有三个分类 需使用One-Hot Encoding 转换
'''
'''
#print(all_df.isnull().sum())
df=all_df.drop(['name'],axis=1)
age_mean=df['age'].mean()
df['age'].fillna(age_mean)

age_mean=df['fare'].mean()
df['fare'].fillna(age_mean)
df['sex']=df['sex'].map({'female':0,'male':1}).astype(int)

#print(df[:2])#数据展示
x_onehot_df=pd.get_dummies(data=df,columns=['embarked'])
print(x_onehot_df[:2])


#将data_frame转变为array
ndarray=x_onehot_df.values
print(ndarray)
#第一个字段是label，后面的是features  提取features和label
Label = ndarray[:,0]
Features = ndarray[:,1:]

#使用preprocessing.MinMaxScaler将ndarray特征字段标准化
minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
scaledFeatures=minmax_scale.fit_transform(Features)
'''
#将数据分为训练集和测试集,按照8：2的比例，使用numpy.random.rand产生msk
msk=np.random.rand(len(all_df))<0.8
train_df=all_df[msk]
test_df=all_df[~msk]
print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))

#定义数据处理函数==========================

def ProcessingData(raw_df):
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age']=df['age'].fillna(age_mean)
    age_mean = df['fare'].mean()
    df['fare']=df['fare'].fillna(age_mean)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    x_onehot_df = pd.get_dummies(data=df, columns=['embarked'])
    ndarray = x_onehot_df.values
    Label = ndarray[:, 0]
    Features = ndarray[:, 1:]
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)
    return scaledFeatures,Label

train_Features,train_Lables=ProcessingData(train_df)
test_Features,test_Lables=ProcessingData(test_df)
print()