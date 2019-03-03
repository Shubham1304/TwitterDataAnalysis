#!/usr/bin/env python
# coding: utf-8

# In[42]:


from pandas import read_csv
from datetime import datetime
from sklearn.metrics import mean_squared_error
import pandas
import math 


# In[2]:


from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('DataSet.csv', header=0, index_col=0)
dataset.values
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    #pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=1, loc='right')
    i += 1
    pyplot.show()
'''for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(dataset[:, group])
    pyplot.title(dataset.columns[group], y=1, loc='right')
    i += 1
pyplot.show()'''
print(values)


# In[3]:


values[1]


# In[4]:


print (dataset.head())


# In[5]:


import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pandas.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('DataSet.csv', header=0, index_col=0)
print(dataset)
values = dataset.values
print(values)
# integer encode direction
encoder = LabelEncoder()
values[:,2] = encoder.fit_transform(values[:,2])
print(values)
# ensure all data is float
values = values.astype('float32')
# normalize features

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[5,6,7,]], axis=1, inplace=True)
print(reframed.head())
print(float(values[2][0]))
print(float(scaled[2][0]))


# In[6]:


print(type(dataset))
print((values))
print (float(values[0][0]))
type(values)


# In[7]:


dataset.values


# In[8]:


dataset.head()


# In[33]:


# split into train and test sets
from sklearn.model_selection import train_test_split
values = reframed.values
X=values[:, :-1]
y=values[:, -1]

train_X, test_X, train_y , test_y = train_test_split(X, y, test_size=0.33, random_state=42)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[10]:


test_X.shape


# In[11]:


from tensorflow import keras as k
#from keras.utils.visualize_util import plot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
# design network
model = Sequential()

model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
#model.add(Flatten())
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, validation_split=0.33,epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[34]:


# make a prediction

yhat = model.predict(test_X)
print(yhat)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = pandas.concat((pandas.DataFrame(yhat), pandas.DataFrame(test_X[:, 1:])), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = pandas.concat((pandas.DataFrame(test_y), pandas.DataFrame(test_X[:, 1:])), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print ((inv_y))
print(inv_yhat)

print('Test RMSE: %.3f' % rmse)


# In[41]:


#print(list(map(float,yhat)))
print(float(yhat[0]))
print (inv_y)
print(inv_yhat)


# In[ ]:





# In[ ]:


from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

inv_y = list(map(int,inv_y))
inv_yhat = list(map(int,inv_yhat))
conf_mat = confusion_matrix(inv_y, inv_yhat)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=dataset['retweet_count'].values, yticklabels=dataset['retweet_count'].values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




