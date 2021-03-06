#!/usr/bin/env python
# coding: utf-8

# # Forecasting Coca-Cola sales based on model based methods and Smoothing techniques

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


# Load dataset
df=pd.read_excel("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 18_Forecasting\\CocaCola_Sales_Rawdata.xlsx")
df


# In[3]:


df.shape


# In[4]:


df[df.duplicated()]


# In[5]:


df.info()


# In[6]:


df.Sales.plot()


# In[10]:


df.hist()


# In[11]:


df.plot(kind='kde')


# # Data Preprocessing

# In[12]:


df['quarter']=0
for i in range(42):
    p=df['Quarter'][i]
    df['quarter'][i]=p[0:2]


# In[13]:


df


# In[31]:


sns.boxplot('quarter','Sales', data=df)


# In[14]:


# Creating dummies for 'quarter' column
dummies = pd.get_dummies(df['quarter'])
dummies


# In[15]:


data=pd.concat([df,dummies],axis=1)
data


# In[16]:


# Calculating column 't', 't_squared' and 'log_sales'
data['t']=np.arange(1,43)
data['t_squared']= data['t']*data['t']
data['log_sales']=np.log(data['Sales'])
data


# In[17]:


train=data.head(30)
test=data.tail(12)


# In[ ]:





# # Calculating RMSE through model based methods

# In[18]:


import statsmodels.formula.api as smf


# In[19]:


# linear model
lin_model = smf.ols("Sales~t",data=train).fit()
lin_pred = pd.Series(lin_model.predict(test['t']))
rmse_lin = np.sqrt(np.mean((np.array(test['Sales'])-np.array(lin_pred))**2))
rmse_lin


# In[20]:


# exponential model
exp_model= smf.ols("log_sales~t", data=train).fit()
exp_pred= pd.Series(exp_model.predict(test['t']))
rmse_exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(exp_pred)))**2))
rmse_exp


# In[21]:


# quadratic model
quad_model= smf.ols("Sales~t+t_squared",data=train).fit()
quad_pred = pd.Series(quad_model.predict(test[['t','t_squared']]))
rmse_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(quad_pred))**2))
rmse_quad


# In[22]:


# Additive seasonality
add_sea_model= smf.ols("Sales~Q1+Q2+Q3", data=train).fit()
add_sea_pred = add_sea_model.predict(test[['Q1','Q2','Q3']])
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(add_sea_pred))**2))
rmse_add_sea


# In[23]:


# Multiplicative seasonality
mul_sea_model=smf.ols("log_sales~Q1+Q2+Q3", data=train).fit()
mul_sea_pred= pd.Series(mul_sea_model.predict(test[['Q1','Q2','Q3']]))
rmse_mul_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(mul_sea_pred)))**2))
rmse_mul_sea


# In[24]:


# Additive seasonality with quadratic trend
add_sea_quad_model= smf.ols("Sales~t+t_squared+Q1+Q2+Q3", data=train).fit()
add_sea_quad_pred=  pd.Series(add_sea_quad_model.predict(test[['t','t_squared','Q1','Q2','Q3']]))
rmse_add_sea_quad=   np.sqrt(np.mean((np.array(test['Sales'])-np.array(add_sea_quad_pred))**2))
rmse_add_sea_quad


# In[25]:


# Multiplicative additive seasonality
mul_add_sea_model=smf.ols("log_sales~t+Q1+Q2+Q3", data=train).fit()
mul_add_sea_pred= pd.Series(mul_add_sea_model.predict(test[['t','Q1','Q2','Q3']]))
rmse_mul_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(mul_add_sea_pred)))**2))
rmse_mul_add_sea


# ### Before applying the ARIMA model, looking if the dataset is stationary or not through Augmented Dickey Fuller Test (although the plot in the starting suggests it is not stationary )

# Null Hypothesis (H0): The series is not stationary
# 
# Alternate Hypothesis (H1): The series is stationary

# In[26]:


from statsmodels.tsa.stattools import adfuller


# In[28]:


adf_test = adfuller(df['Sales'])
adf_test


# In[29]:


print('ADF stats:', adf_test[0])
print('p-value:',adf_test[1])


# ### Since p>0.05, hence, we fail to reject the null hypothesis. Thus, we can say that the time series is not stationary.So in order to apply ARIMA model, series need to be stationary which it will take care of through differencing. 

# In[21]:


# ARIMA Model


# In[23]:


arima = pd.read_excel("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 18_Forecasting\\CocaCola_Sales_Rawdata.xlsx",
                     header=0,index_col=0, parse_dates=True)
arima


# In[26]:


split_point = len(arima) - 12
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


X = train.values
X


# In[30]:


X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]


# In[31]:


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


# ### Grid search for p,d,q values

# In[32]:


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


# In[33]:


# load dataset
train = read_csv('dataset_cc.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# evaluate parameters
p_values = range(0, 5)
d_values = range(0, 5)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
evaluate_models(train.values, p_values, d_values, q_values)


# ### Build model on optimized values

# In[34]:


# load data
train = read_csv('dataset_cc.csv', header=0, index_col=0, parse_dates=True)
# prepare data
X = train.values
X = X.astype('float32')


# In[36]:


# fit model
model = ARIMA(X, order=(3,2,0))
model_fit = model.fit()
forecast=model_fit.forecast(steps=12)[0]
model_fit.plot_predict(2,30)


# In[38]:


#Error on the test data
val=pd.read_csv('validation_cc.csv',header=None)
rmse_ARIMA = np.sqrt(mean_squared_error(val[1], forecast))
rmse_ARIMA


# In[39]:


table={"Model":["Linear","Exponential","Quadratic","Additive Seasonality","Multiplicative Seasonality",
                "Additive Seasonality with quadratic trend","Multiplicative Additive Seasonality","ARIMA"],
       "RMSE values":[rmse_lin,rmse_exp,rmse_quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea,rmse_ARIMA]}
frame= pd.DataFrame(table)
frame


# In[40]:


frame.sort_values("RMSE values")


# # RMSE based on Smoothing techniques

# In[33]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


# In[34]:


graph = seasonal_decompose(df['Sales'],period=4)
graph.plot()
plt.show()


# In[35]:


import statsmodels.graphics.tsaplots as tsa_plot
tsa_plot.plot_acf(df['Sales'],lags=12)
tsa_plot.plot_pacf(df['Sales'],lags=12)


# In[36]:


Train=df.head(30)
Test=df.tail(12)


# In[37]:


# Simple Exponential Smoothing
level=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for i in level:
    ses = SimpleExpSmoothing(Train['Sales']).fit(smoothing_level=i)
    ses_pred = ses.predict(start=Test.index[0],end=Test.index[-1])
    rmse_ses= np.sqrt(np.mean((np.array(Test['Sales'])-np.array(ses_pred))**2))
    print("RMSE for smoothing level {} is {}".format(i,rmse_ses))


# ### We get lowest RMSE of 667.88 at smoothing level = 1 for Simple exponential smoothing

# In[38]:


# Holt Method
level= [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
slope= [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

for i in level:
    for y in slope:
        hw_model = Holt(Train["Sales"]).fit(smoothing_level=i, smoothing_slope=y)
        pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
        rmse_hw= np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_hw))**2))
        print("RMSE for smoothing level {} and smoothing slope {} is {}".format(i,y,rmse_hw))


# ### Thus, in Holt's method, we get least RMSE of 394.84 when smoothing level is 0.3 and smoothing slope is 0.9

# In[39]:


# Holt winter exponential smoothing with additive trend and additive seasonality
hw_add = ExponentialSmoothing(Train['Sales'],trend='add',seasonal='add',seasonal_periods=4).fit()
hw_add_pred = hw_add.predict(start=Test.index[0],end=Test.index[-1])
rmse_hw_add= np.sqrt(np.mean((np.array(Test['Sales'])-np.array(hw_add_pred))**2))
rmse_hw_add


# In[40]:


# Holt winter exponential smoothing with additive trend and multiplicative seasonality
hw_mul = ExponentialSmoothing(Train['Sales'],trend='add',seasonal='mul',seasonal_periods=4).fit()
hw_mul_pred = hw_mul.predict(start=Test.index[0],end=Test.index[-1])
rmse_hw_mul = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(hw_mul_pred))**2))
rmse_hw_mul


# In[46]:


table1={"Data driven model": ["SES","Holt","HW_Add","HW_Mul"], "RMSE Values":[667.88,394.84,rmse_hw_add,rmse_hw_mul]}
frame1=pd.DataFrame(table1)
frame1


# In[47]:


frame1.sort_values("RMSE Values")


# ### Thus, we see that out of model based approach and data driven approach, Holtwinter model with additive trend and multiplicative seasonality has the least RMSE of 181.71. This is the model that will be the final choice.

# In[43]:


# Final model
hw_mul_whole = ExponentialSmoothing(df['Sales'],trend='add',seasonal='mul',seasonal_periods=4).fit()


# In[44]:


pred=hw_mul_whole.forecast(10)


# In[45]:


pred


# In[ ]:




