#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Load the dataset
data=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 16_Neural Networks\\gas_turbines.csv")
data


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data[data.duplicated()]


# In[6]:


# Dividing the data into X & Y
X=data.drop('TEY',axis=1)
X


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps


# In[8]:


list=['AT','AP','AH','AFDP','GTEP','TIT','TAT','CDP','CO','NOX']
score=[]
for i in list:
    z=pps.score(data,i,"TEY")
    print('PPS score for {} with TEY is {}'. format(i,z['ppscore']))
    score.append(z['ppscore'])

plt.bar(range(len(list)),score)
plt.xlabel("Gas Turbine Features")
plt.ylabel("Power Predicted Score(PPS)")
plt.show()


# ### Thus, we can see that few of the features have a PPS of zero. Dropping these features.

# In[9]:


X=X.drop(['AT','AP','AH','NOX'],axis=1)
X


# In[11]:


# Checking for outliers in 'X' data
fig, ax = plt.subplots(2,3, figsize=(15,10))

sns.boxplot(X.AFDP, ax=ax[0,0])
sns.boxplot(X.GTEP, ax=ax[0,1])
sns.boxplot(X.TIT, ax=ax[0,2])
sns.boxplot(X.TAT, ax=ax[1,0])
sns.boxplot(X.CDP, ax=ax[1,1])
sns.boxplot(X.CO, ax=ax[1,2])

plt.tight_layout()
plt.show()


# ### Although there are some outliers in the data but they don't look like error values but values captured naturally. So retaining the outliers for model training.

# In[12]:


Y=data['TEY']
Y


# In[13]:


y_new=pd.DataFrame(Y, columns=['TEY'])
y_new


# In[14]:


# Normalizing the data
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x = scale.fit_transform(X)
y = scale.fit_transform(y_new)


# In[15]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np


# In[16]:


np.random.seed(7)
from sklearn.metrics import mean_squared_error


# In[17]:


# create model
model = Sequential()
model.add(Dense(5, input_dim=6,  activation='LeakyReLU')) #1st layer
model.add(Dropout(0.3))
model.add(Dense(3, activation='LeakyReLU')) #2nd layer
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear')) #3rd layer or top layer

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['MeanSquaredError'])


# In[18]:


# Fit the model
history=model.fit(x, y, validation_split=0.33, epochs=250, batch_size=8)


# In[21]:


# evaluate the model
scores = model.evaluate(x, y)
print(scores)
print("The {} for the test model is {}".format(model.metrics_names[1], scores[1]))


# ### The model has a mean squared error of 0.069

# In[31]:


history.history


# In[32]:


# Visualize training history

# list all data in history
model.history.history.keys()


# In[33]:


history.history.keys()


# In[34]:


# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model error')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # Checking for mean square error after k-fold cross validation

# In[27]:


from sklearn.model_selection import KFold


# In[29]:


kfold= KFold(n_splits=5, random_state=7, shuffle = True)
msescores=[]
for train, test in kfold.split(x,y):
    model = Sequential()
    model.add(Dense(5, input_dim=6,  activation='LeakyReLU')) 
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='LeakyReLU'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['MeanSquaredError'])
    model.fit(x[train], y[train], epochs=150, batch_size=8,verbose=0)
    scores = model.evaluate(x[test], y[test],verbose=0)
    print("The {} for the model is {}".format(model.metrics_names[1], scores[1]))
    msescores.append(scores[1])
    print(msescores)
    


# In[30]:


print(np.mean(msescores))


# ### Thus, we see that the mean squared error after K-fold cross validation comes out to be 0.054 which is lower than 0.069 obtained above.

# In[ ]:




