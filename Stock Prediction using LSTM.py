#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
        


# In[3]:


df=pd.read_csv('NFLX.csv')


# # READING DATA
# 

# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[9]:


df.info()


# In[10]:


df.columns


# In[12]:


df.isnull()


# In[13]:


df.isnull().sum()


# We see there is no null value in the dataset

# In[14]:


df=df[['Date', 'Close','Open']]
df


# In[15]:


# converting object dtype of date column to datetime dtype
df['Date'] = pd.to_datetime(df['Date'])
df['Date']


# In[16]:


df.set_index('Date',drop=True,inplace=True)


# In[17]:


df.head()


# In[18]:


df.dtypes


# In[21]:



sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(25, 7))

# Plot 'open' column
sns.lineplot(data=df, x=df.index, y='Open', ax=ax[0], label='Open', color='green')
ax[0].set_xlabel('Date', size=20)
ax[0].set_ylabel('Price', size=20)
ax[0].legend()

# Plot 'close' column
sns.lineplot(data=df, x=df.index, y='Close', ax=ax[1], label='Close', color='orange')
ax[1].set_xlabel('Date', size=20)
ax[1].set_ylabel('Price', size=20)
ax[1].legend()

plt.show()


# DATA PRE PROCESSING

# In[22]:


print(df.dtypes)


# In[23]:


scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df)
df


# In[24]:


training_size= int(len(df) * 0.70)
#use 70 % data for training and the rest 30% for testing
training_size


# In[25]:


train_data = df[:training_size]
test_data  = df[training_size:]
train_data.shape, test_data.shape


# In[26]:


# Function to create sequence of data for training and testing

def create_sequence(dataset):
    sequences = []
    labels = []
    start_idx = 0

    for stop_idx in range(50,len(dataset)):
        # Selecting 50 rows at a time
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1
    return (np.array(sequences),np.array(labels))


# In[27]:


train_seq, train_label = create_sequence(train_data) 
test_seq, test_label = create_sequence(test_data)
train_seq.shape, train_label.shape, test_seq.shape, test_label.shape


# # IMPLEMENTING LSTM 

# In[28]:


# imported Sequential from keras.models  
model = Sequential()
# importing Dense, Dropout, LSTM, Bidirectional from keras.layers 
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))
model.add(Dropout(0.1)) 
model.add(LSTM(units=50))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()


# In[29]:


model.fit(train_seq, train_label, epochs=100,validation_data=(test_seq, test_label), verbose=1)


# In[30]:


# predicting the values after running the model
test_predicted = model.predict(test_seq)
test_predicted[:10]


# In[31]:


# Inversing normalization/scaling on predicted data 
test_inverse_predicted = scaler.inverse_transform(test_predicted)
test_inverse_predicted[:5]


# In[32]:


# Inversing normalization/scaling on predicted data 
test_inverse_predicted = scaler.inverse_transform(test_predicted)
test_inverse_predicted[:5]


# 
# # VISUALIZING ACTUAL VS PREDICTED DATA

# In[36]:


# Merging actual and predicted data for better visualization
df_merge = pd.concat([df.iloc[-253:].copy(), pd.DataFrame(test_inverse_predicted[-253:], columns=['open_predicted', 'close_predicted'], index=df.iloc[-253:].index)], axis=1)


# In[37]:


df_merge[['Open','Close']] = scaler.inverse_transform(df_merge[['Open','Close']])
df_merge.head()


# In[38]:


# plotting the actual open and predicted open prices on date index
df_merge[['Open','open_predicted']].plot(figsize=(8,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for open price',size=15)
plt.show()


# In[39]:


# plotting the actual close and predicted close prices on date index 
df_merge[['Close','close_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for close price',size=15)
plt.show()


# In[40]:


# Creating a dataframe and adding 15 days to existing index 

df_merge = df_merge.append(pd.DataFrame(columns=df_merge.columns,
                                        index=pd.date_range(start=df_merge.index[-1], periods=11, freq='D', closed='right')))
df_merge['2021-06-09':'2021-06-21']


# In[41]:


# creating a DataFrame and filling values of open and close column
upcoming_prediction = pd.DataFrame(columns=['open','close'],index=df_merge.index)
upcoming_prediction.index=pd.to_datetime(upcoming_prediction.index)


# In[42]:


curr_seq = test_seq[-1:]

for i in range(-10,0):
  up_pred = model.predict(curr_seq)
  upcoming_prediction.iloc[i] = up_pred
  curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
  curr_seq = curr_seq.reshape(test_seq[-1:].shape)


# In[43]:


upcoming_prediction[['open','close']] = scaler.inverse_transform(upcoming_prediction[['open','close']])
# plotting Upcoming Open price on date index
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_merge.loc['2021-04-01':,'Open'],label='Current Open Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'open'],label='Upcoming Open Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming Open price prediction',size=15)
ax.legend()
fig.show()


# In[44]:


# plotting Upcoming Close price on date index
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_merge.loc['2021-04-01':,'Close'],label='Current close Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'close'],label='Upcoming close Price',)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming close price prediction',size=15)
ax.legend()
fig.show()


# In[ ]:




