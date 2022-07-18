#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


get_ipython().system('pip install keras')


# In[1]:


import pandas as pd


# In[2]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy 


# In[3]:


numpy.random.seed(7)
# Load Dataset
data=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 16_Neural Networks\\forestfires.csv")
data


# In[4]:


data.info()


# In[5]:


data[data.duplicated()]


# In[6]:


# Dropping duplicate rows
data1= data.drop_duplicates()


# In[7]:


data1.shape


# In[8]:


# Dropping columns 10 to 29
data1 = data1.drop(data.columns[10:30],axis=1)
data1


# In[17]:


# Checking for class counts in categorical variables
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data1['month'])
plt.show()
sns.countplot(data1['day'])
plt.show()
sns.countplot(data1['size_category'])
plt.show()


# ### Thus we see that in month column ' aug' and 'sep' values dominate whereas in day column the count of different days is fairly proportional. Similarly, there is a class imbalance in 'size_category'

# In[16]:


# Checking of outlier data in numerical columns

fig, ax = plt.subplots(3,3, figsize=(15,10))

sns.boxplot(data1.FFMC, ax=ax[0,0])
sns.boxplot(data1.DMC, ax=ax[0,1])
sns.boxplot(data1.DC, ax=ax[0,2])
sns.boxplot(data1.ISI, ax=ax[1,0])
sns.boxplot(data1.temp, ax=ax[1,1])
sns.boxplot(data1.RH, ax=ax[1,2])
sns.boxplot(data1.wind, ax=ax[2,0])
sns.boxplot(data1.rain, ax=ax[2,1])
plt.tight_layout()
plt.show()


# ### Although there are some outliers in the data but they don't look like error values but values captured naturally. So retaining the outliers for model training.

# In[9]:


data_norm =['FFMC','DMC','DC','ISI','temp','RH','wind','rain']
data_norm


# In[10]:


# Normalizing the numerical data only in data1
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scaled_data=scale.fit_transform(data1[data_norm])
scaled_data


# In[11]:


scaled_data.shape


# In[12]:


dataframe_normalized = pd.DataFrame(scaled_data,columns=['FFMC','DMC','DC','ISI','temp','RH','wind','rain'])
dataframe_normalized


# In[13]:


# Dropping the numerical columns from the data1
data1 = data1.drop(data1.columns[2:10],axis=1)
data1


# In[14]:


data1=data1.reset_index()
data1


# In[15]:


data1=data1.drop('index',axis=1)
data1


# In[16]:


# Changing size_category's categorical values to numerical values
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
data1.iloc[:,2]=labelencoder.fit_transform(data1.iloc[:,2])
data1.head()


# In[17]:


data1['month'].replace(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                    [1,2,3,4,5,6,7,8,9,10,11,12], inplace = True)


# In[18]:


data1['day'].replace(['mon','tue','wed','thu','fri','sat','sun'],[1,2,3,4,5,6,7],inplace = True)


# In[19]:


data1


# In[20]:


# Concatinating the categorical and numerical dataframes together for analysis
df=pd.concat([dataframe_normalized,data1],axis=1)
df


# In[21]:


X=df.iloc[:,0:10]
Y=df.iloc[:,10]


# In[22]:


# create model
model = Sequential()
model.add(Dense(7, input_dim=10,  activation='LeakyReLU')) #1st layer
model.add(Dropout(0.3))
model.add(Dense(4, activation='LeakyReLU')) #2nd layer
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid')) #3rd layer or top layer

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[23]:


# Fit the model
history=model.fit(X, Y, validation_split=0.33, epochs=250, batch_size=8)


# In[24]:


# evaluate the model
scores = model.evaluate(X, Y)
print(scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[25]:


history.history


# In[26]:


# Visualize training history

# list all data in history
model.history.history.keys()


# In[27]:


history.history.keys()


# In[28]:


# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
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


# ### Thus we see that accuracy is less owing to the increasing model loss with accuracy at 67.5%

# # Using Stratified KFold technique to improve accuracy by taking into account class imbalance for 'Y'

# In[54]:


from sklearn.model_selection import StratifiedKFold


# In[57]:


kfold= StratifiedKFold(n_splits=5, random_state=7, shuffle = True)
cvscores=[]
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(7, input_dim=10,  activation='LeakyReLU')) 
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='LeakyReLU'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X.iloc[train], Y.iloc[train], epochs=150, batch_size=8,verbose=0)
    scores = model.evaluate(X.iloc[test], Y.iloc[test],verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    print(cvscores)
    


# In[58]:


print(numpy.mean(cvscores))


# ### Thus we see that the accuracy of the model has increased from 67.5% to 72.5% using stratified KFold technique 

# In[ ]:




