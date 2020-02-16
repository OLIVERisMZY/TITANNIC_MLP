from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
import  pandas as pd
from 数据处理 import all_df,test_Features,test_Lables,train_Features,train_Lables,ProcessingData
#建立的多层感知机模型包含：输入层（9个神经元），隐含层1（40个units），隐含层2（30个units）
model=Sequential()
model.add(Dense(units=40,input_dim=9,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=30,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=train_Features,y=train_Lables,validation_split=0.1,epochs=30,batch_size=30,verbose=2)

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
#show_train_history(train_history,'acc','val_acc')
scores=model.evaluate(x=test_Features,y=test_Lables)
print('最终得分为：'+str(scores[1]))


#加入杰克罗斯数据进行预测=========
#使用pd.DataFrame()创建DataFrame 加入jack和rose的数据
Jack=pd.Series([0,'Jack',3,'male',23,1,0,5.0000,'S'])
Rose=pd.Series([1,'Rose',1,'female',20,1,0,100.0000,'S'])
JR_df=pd.DataFrame([list(Jack),list(Rose)],columns=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked'])
all_df=pd.concat([all_df,JR_df])
all_Features,Lable=ProcessingData(all_df)
all_probability=model.predict(all_Features)
print(all_probability[:10])
#将all_probability数据传入all_df 进行整合，产生pd DataFrame.查看后两项数据
pd=all_df
pd.insert(len(all_df.columns),'probability',all_probability)
print(pd[-2:])