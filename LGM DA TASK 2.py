#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('C:\\Users\\Harsh\\Desktop\\lgm da\\data.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df1=df.reset_index()['Close']


# In[6]:


df1


# In[7]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[8]:


import numpy as np


# In[9]:


df1


# In[10]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[11]:


print(df1)


# In[12]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[13]:


training_size,test_size


# In[14]:


train_data


# In[15]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[16]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[17]:


print(X_train.shape), print(y_train.shape)


# In[18]:


print(X_test.shape), print(ytest.shape)


# In[19]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[21]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[22]:


model.summary()


# In[23]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[24]:


import tensorflow as tf


# In[25]:


tf.__version__


# In[26]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[27]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[28]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[29]:


math.sqrt(mean_squared_error(ytest,test_predict))


# In[30]:


look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[31]:


len(test_data)


# In[32]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[33]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[34]:


temp_input


# In[42]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[43]:


import matplotlib.pyplot as plt


# In[44]:


len(df1)

