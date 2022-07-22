#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[62]:


# Load dataset
data=pd.read_excel("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 18_Forecasting\\Airlines+Data.xlsx")
data


# In[63]:


data[data.duplicated()]


# In[64]:


data.info()


# In[5]:


data['Passengers'].plot()


# In[3]:


# Extracting 'month' and 'year' column from 'Month'
data['month']= data['Month'].dt.strftime("%b")
data['year']= data['Month'].dt.strftime("%Y")
data


# In[7]:


# Getting dummies for column 'month' and concatinating with 'data' dataframe
dummy= pd.get_dummies(data['month'])
df=pd.concat([data,dummy],axis=1)
df


# In[9]:


# Creating columns 't', 't_squared' and 'log_pass'
df['t']=np.arange(1,97)
df['t_squared'] = df['t']**2
df['log_pass']=np.log(df['Passengers'])
df


# In[11]:


plt.figure(figsize=(10,6))
sns.boxplot(df['month'],df['Passengers'])


# In[12]:


sns.lineplot(df['year'],df['Passengers'])


# In[13]:


train=df.head(80)
test=df.tail(16)


# # Calculating RMSE from model based methods

# In[14]:


# linear model
import statsmodels.formula.api as smf
lin_model= smf.ols('Passengers~t',data= train).fit()
lin_pred= pd.Series(lin_model.predict(test['t']))
rmse_lin = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(lin_pred))**2))
rmse_lin


# In[15]:


# Exponential model
exp_model = smf.ols('log_pass~t',data=train).fit()
exp_pred = pd.Series(exp_model.predict(test['t']))
rmse_exp = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(exp_pred)))**2))
rmse_exp


# In[16]:


# Quadratic model
quad_model = smf.ols('Passengers~t+t_squared',data=train).fit()
quad_pred = pd.Series(quad_model.predict(test[['t','t_squared']]))
rmse_quad= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(quad_pred))**2))
rmse_quad


# In[17]:


# Additive Seasonality
add_sea_model = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
add_sea_pred=pd.Series(add_sea_model.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(add_sea_pred))**2))
rmse_add_sea


# In[19]:


# Multiplicative Seasonality
mul_sea_model=smf.ols('log_pass~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
mul_sea_pred = pd.Series(mul_sea_model.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_mul_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(mul_sea_pred)))**2))
rmse_mul_sea


# In[21]:


# Additive seasonality with quadratic trend
add_sea_quad_model = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
add_sea_quad_pred=pd.Series(add_sea_quad_model.predict(test[['t','t_squared','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(add_sea_quad_pred))**2))
rmse_add_sea_quad


# In[22]:


# Multiplicative additive seasonality
mul_add_sea_model = smf.ols('log_pass~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
mul_add_sea_pred = pd.Series(mul_add_sea_model.predict(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_mul_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(mul_add_sea_pred)))**2))
rmse_mul_add_sea


# In[23]:


# ARIMA model
arima = pd.read_excel("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 18_Forecasting\\Airlines+Data.xlsx",
                     header=0,index_col=0, parse_dates=True)
arima


# In[24]:


sns.histplot(arima['Passengers'])


# In[25]:


sns.distplot(arima['Passengers'])


# In[26]:


split_point = len(arima) - 16
dataset_cc, validation_cc = arima[0:split_point], arima[split_point:]
print('Dataset_cc %d, Validation_cc %d' % (len(dataset_cc), len(validation_cc)))


# In[27]:


dataset_cc.to_csv('dataset_cc.csv', header=False)
validation_cc.to_csv('validation_cc.csv', header=False)


# In[28]:


from pandas import read_csv
train = read_csv('dataset_cc.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
train


# In[29]:


X=train.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]


# In[30]:


from sklearn.metrics import mean_squared_error
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse_arima = np.sqrt(mean_squared_error(test, predictions))
print('RMSE_ARIMA: %.3f' % rmse_arima)


# In[34]:


table={'Model':['Linear','Exponential','Quadratic','Additive Seasonality','Multiplicative Seasonality',
    'Add Seasonality with quad trend','Multiplicative additive Seasonality','ARIMA'],'RMSE Values':[rmse_lin,rmse_exp,
    rmse_quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad, rmse_mul_add_sea,rmse_arima]}
frame=pd.DataFrame(table)
frame


# In[35]:


frame.sort_values("RMSE Values")


# # RMSE based on data driven models

# In[37]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


# In[39]:


graph= seasonal_decompose(data['Passengers'],period=12)
graph.plot()
plt.show()


# In[40]:


import statsmodels.graphics.tsaplots as tsa_plot


# In[42]:


tsa_plot.plot_acf(data['Passengers'],lags=12)
tsa_plot.plot_pacf(data['Passengers'],lags=12)


# In[43]:


Train=data.head(80)
Test = data.tail(16)


# In[45]:


# Simple Exponential Smoothing
level=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for i in level:
    ses = SimpleExpSmoothing(Train['Passengers']).fit(smoothing_level=i)
    ses_pred = ses.predict(start=Test.index[0],end=Test.index[-1])
    rmse_ses= np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(ses_pred))**2))
    print("RMSE for smoothing level {} is {}".format(i,rmse_ses))


# ### Thus, we get lowest RMSE of 47.73 with smoothing_level=0.3 in Simple Exponential Smoothing model

# In[52]:


# Holt method
level= [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
slope= [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

for i in level:
    for y in slope:
        hw_model = Holt(Train['Passengers']).fit(smoothing_level=i,smoothing_slope=y)
        hw_pred= hw_model.predict(start=Test.index[0],end=Test.index[-1])
        rmse_hw= np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(hw_pred))**2))
        print("RMSE for smoothing level {} and smoothing slope {} is {}".format(i,y,rmse_hw))


# ### Thus, for Holt's method we get least RMSE of 43.38 at smoothing level=0.1 and smoothing slope=0.1

# In[53]:


# Exponential Smoothing with additive trend and additive seasonality
exp_smoothing= ExponentialSmoothing(Train['Passengers'], trend='add',seasonal='add',seasonal_periods=12).fit()
exp_smoothing_pred = exp_smoothing.predict(start=Test.index[0],end=Test.index[-1])
rmse_exp_smoothing_add = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(exp_smoothing_pred))**2))
rmse_exp_smoothing_add


# In[54]:


# Exponential smoothing with additive trend and multiplicative seasonality
exp_smoothing_mul = ExponentialSmoothing(Train['Passengers'], trend='add',seasonal='mul',seasonal_periods=12).fit()
exp_smoothing_mul_pred = exp_smoothing_mul.predict(start=Test.index[0],end=Test.index[-1])
rmse_exp_smoothing_mul = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(exp_smoothing_mul_pred))**2))
rmse_exp_smoothing_mul


# In[55]:


table1={"Data driven model": ["SES","Holt","HW_Add","HW_Mul"], "RMSE Values":[47.73,43.38,rmse_exp_smoothing_add,
                                                                             rmse_exp_smoothing_mul]}
frame1=pd.DataFrame(table1)
frame1


# In[56]:


frame1.sort_values("RMSE Values")


# ### Hence, we can see that out of all the models discussed above multiplicative additive seasonality has the least RMSE of 9.47. Therefore, we will apply this model on the whole dataset to get the forecast.

# In[ ]:




