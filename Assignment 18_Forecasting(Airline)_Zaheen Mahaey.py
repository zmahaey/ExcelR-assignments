#!/usr/bin/env python
# coding: utf-8

# # Calculating RMSE for forecasting models using model based methods and smoothing techniques

# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


# Load dataset
data=pd.read_excel("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 18_Forecasting\\Airlines+Data.xlsx")
data


# In[3]:


data[data.duplicated()]


# In[4]:


data.info()


# In[5]:


data['Passengers'].plot()


# In[6]:


data['Passengers'].hist()


# In[7]:


data['Passengers'].plot(kind='kde')


# In[8]:


# Extracting 'month' and 'year' column from 'Month'
data['month']= data['Month'].dt.strftime("%b")
data['year']= data['Month'].dt.strftime("%Y")
data


# In[9]:


# Getting dummies for column 'month' and concatinating with 'data' dataframe
dummy= pd.get_dummies(data['month'])
df=pd.concat([data,dummy],axis=1)
df


# In[10]:


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


# In[18]:


# Multiplicative Seasonality
mul_sea_model=smf.ols('log_pass~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
mul_sea_pred = pd.Series(mul_sea_model.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_mul_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(mul_sea_pred)))**2))
rmse_mul_sea


# In[19]:


# Additive seasonality with quadratic trend
add_sea_quad_model = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
add_sea_quad_pred=pd.Series(add_sea_quad_model.predict(test[['t','t_squared','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(add_sea_quad_pred))**2))
rmse_add_sea_quad


# In[20]:


# Multiplicative additive seasonality
mul_add_sea_model = smf.ols('log_pass~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
mul_add_sea_pred = pd.Series(mul_add_sea_model.predict(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_mul_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(mul_add_sea_pred)))**2))
rmse_mul_add_sea


# ### Before applying the ARIMA model, looking if the dataset is stationary or not through Augmented Dickey Fuller Test (although the plot in the starting suggests it is not stationary )

# Null Hypothesis (H0): The series is not stationary
# 
# Alternate Hypothesis (H1): The series is stationary

# In[43]:


from statsmodels.tsa.stattools import adfuller


# In[44]:


adf_test = adfuller(data['Passengers'])
adf_test


# In[45]:


print('ADF stats:', adf_test[0])
print('p-value:',adf_test[1])


# ### Since p>0.05, hence, we fail to reject the null hypothesis. Thus, we can say that the time series is not stationary.So in order to apply ARIMA model, series need to be stationary which it will take care of through differencing. 

# In[ ]:


# ARIMA model
arima = pd.read_excel("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 18_Forecasting\\Airlines+Data.xlsx",
                     header=0,index_col=0, parse_dates=True)
arima


# In[22]:


split_point = len(arima) - 16
dataset_cc, validation_cc = arima[0:split_point], arima[split_point:]
print('Dataset %d, Validation %d' % (len(dataset_cc), len(validation_cc)))


# In[23]:


dataset_cc.to_csv('dataset.csv', header=False)
validation_cc.to_csv('validation.csv', header=False)


# In[24]:


from pandas import read_csv
train = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
train


# ### ARIMA Hyperparameters

# In[26]:


# grid search ARIMA parameters for a time series

import warnings
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
# prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
# make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
# calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# ### Grid search p,d,q values

# In[27]:


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(train, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# In[28]:


# load dataset
train = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# evaluate parameters
p_values = range(0, 5)
d_values = range(0, 5)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
evaluate_models(train.values, p_values, d_values, q_values)


# ### Build model on optimized values

# In[29]:


# load data
train = read_csv('dataset.csv', header=0, index_col=0, parse_dates=True)
# prepare data
X = train.values
X = X.astype('float32')


# In[37]:


# fit model
model = ARIMA(X, order=(0,1,4))
model_fit = model.fit()
forecast=model_fit.forecast(steps=16)[0]
model_fit.plot_predict(1,96)


# In[39]:


#Error on the test data
val=pd.read_csv('validation.csv',header=None)
rmse_ARIMA = np.sqrt(mean_squared_error(val[1], forecast))
rmse_ARIMA


# In[40]:


table={'Model':['Linear','Exponential','Quadratic','Additive Seasonality','Multiplicative Seasonality',
    'Add Seasonality with quad trend','Multiplicative additive Seasonality','ARIMA'],'RMSE Values':[rmse_lin,rmse_exp,
    rmse_quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad, rmse_mul_add_sea,rmse_ARIMA]}
frame=pd.DataFrame(table)
frame


# In[41]:


frame.sort_values("RMSE Values")


# # RMSE based on Smoothing techniques

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




