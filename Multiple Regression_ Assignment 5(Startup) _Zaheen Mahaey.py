#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 5_Multiple regression\\50_Startups.csv")
data


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data[data.duplicated()]


# In[6]:


data = data.drop('State',axis=1)
data.head()


# In[7]:


data=data.rename({"R&D Spend": "res_spend", "Administration": "admin_spend", "Marketing Spend": "mkt_spend"},axis=1)
data.head()


# In[8]:


data.corr()


# In[10]:


sns.set_style(style='darkgrid')
sns.pairplot(data)


# In[13]:


import statsmodels.formula.api as smf
model=smf.ols("Profit~ res_spend+ admin_spend + mkt_spend", data=data).fit()


# In[14]:


model.rsquared


# In[15]:


model.fittedvalues


# In[16]:


model.params


# In[17]:


model.pvalues, model.tvalues


# In[19]:


# Calculating Varinace inflation Factor (VIF)
rsq_res_spend = smf.ols('res_spend ~ admin_spend + mkt_spend',data=data).fit().rsquared
vif_res_spend = 1/(1-rsq_res_spend)

rsq_admin_spend = smf.ols('admin_spend ~ res_spend + mkt_spend', data=data).fit().rsquared
vif_admin_spend = 1/(1-rsq_admin_spend)

rsq_mkt_spend = smf.ols('mkt_spend ~ res_spend + admin_spend', data=data).fit().rsquared
vif_mkt_spend = 1-(1-rsq_mkt_spend)

d={"Variables": ["res_spend","admin_spend", "mkt_spend"], "VIF values": [vif_res_spend,vif_admin_spend,vif_mkt_spend]}
dataframe=pd.DataFrame(d)
dataframe


# Since, none of the VIF values are greater than 20, multicollinearity does not exist 

# # Residual Analysis

# In[18]:


# Normal QQ Plot
import statsmodels.api as sm
qqplot=sm.qqplot(model.resid, line='q')
plt.title("Normal Residual QQ PLot")
plt.show()


# In[20]:


sns.boxplot(model.resid)


# In[21]:


sns.histplot(model.resid)


# In[22]:


sns.distplot(model.resid)


# # Checking normality through Homoscedasticity

# In[23]:


def standardized_values(vals):
    return((vals-vals.mean())/vals.std())


# In[24]:


plt.scatter(standardized_values(model.fittedvalues),
           standardized_values(model.resid))
plt.title("Residual Plot")
plt.xlabel("Standardized Predicted Values")
plt.ylabel("Standardized Residual Values")
plt.show()


# In[25]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"res_spend", fig= fig)
plt.show()


# In[26]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"admin_spend", fig= fig)
plt.show()


# In[27]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"mkt_spend", fig= fig)
plt.show()


# ### From residual analysis and QQ plot we can see that data is not normal and there is fanning in variance. Hence, we will apply transformations.

# # 1) Log Transformation - Since the non normality and variance is in question along with non linearity, we will start by taking log of the dependent variable

# In[29]:


data1=data.copy()
data1.head()


# In[31]:


log_profit = np.log(data1['Profit'])


# In[32]:


data1['log_profit']=log_profit
data1.head()


# In[33]:


model_log = smf.ols("log_profit~ res_spend+ admin_spend + mkt_spend", data=data1).fit()


# In[34]:


model_log.rsquared


# In[35]:


((model_log.resid)**2).mean()


# In[36]:


data1.corr()


# In[37]:


sns.set_style(style='darkgrid')
sns.pairplot(data1)


# In[38]:


# Generating QQ plot
qqplot= sm.qqplot(model_log.resid, line='q')
plt.title("Normal QQ plot of the residuals for model_log")
plt.show()


# In[41]:


sns.boxplot(model_log.resid)


# # Checking normality through Homscedasticity

# In[42]:


plt.scatter(standardized_values(model_log.fittedvalues),
           standardized_values(model_log.resid))
plt.title("Residual Plot")
plt.xlabel("Standardized Predicted Values")
plt.ylabel("Standardized Residual Values")
plt.show()


# In[45]:


plt.scatter(standardized_values(data1['res_spend']),
           standardized_values(model_log.resid))
plt.title("Residual Plot")
plt.xlabel("Standardized Predicted Values")
plt.ylabel("Standardized Residual Values")
plt.show()


# In[46]:


plt.scatter(standardized_values(data1['admin_spend']),
           standardized_values(model_log.resid))
plt.title("Residual Plot")
plt.xlabel("Standardized Predicted Values")
plt.ylabel("Standardized Residual Values")
plt.show()


# In[47]:


plt.scatter(standardized_values(data1['mkt_spend']),
           standardized_values(model_log.resid))
plt.title("Residual Plot")
plt.xlabel("Standardized Predicted Values")
plt.ylabel("Standardized Residual Values")
plt.show()


# ### As it can be observed from the graphical outputs, normality,linearity and variance can still be improved. Hence, we need to go for some other transformation

# # 2) Square root transformation

# In[48]:


data2=data.copy()
data2.head()


# In[50]:


t_profit =np.sqrt(data2['Profit'])


# In[51]:


data2['t_profit']= t_profit
data2.head()


# In[53]:


model_sqrt = smf.ols("t_profit~ res_spend+ admin_spend + mkt_spend", data=data2).fit()


# In[54]:


model_sqrt.rsquared


# In[56]:


data2.corr()


# In[57]:


sns.set_style(style='darkgrid')
sns.pairplot(data2)


# In[58]:


sns.boxplot(model_sqrt.resid)


# In[59]:


sns.histplot(model_sqrt.resid)


# In[60]:


sns.distplot(model_sqrt.resid)


# In[61]:


((model_sqrt.resid)**2).mean()


# In[62]:


model_sqrt.params


# In[63]:


model_sqrt.pvalues


# In[64]:


# Calculating Varinace inflation Factor (VIF)
rsq_res_spend = smf.ols('res_spend ~ admin_spend + mkt_spend',data=data2).fit().rsquared
vif_res_spend = 1/(1-rsq_res_spend)

rsq_admin_spend = smf.ols('admin_spend ~ res_spend + mkt_spend', data=data2).fit().rsquared
vif_admin_spend = 1/(1-rsq_admin_spend)

rsq_mkt_spend = smf.ols('mkt_spend ~ res_spend + admin_spend', data=data2).fit().rsquared
vif_mkt_spend = 1-(1-rsq_mkt_spend)

d1 = {'Variables': ['res_spend','admin_spend','mkt_spend'], 'VIF':[vif_res_spend,vif_admin_spend,vif_mkt_spend]}
dataframe1 = pd.DataFrame(d1)
dataframe1


# # Residual Analysis

# In[65]:


# Generating QQ plot
qqplot= sm.qqplot(model_sqrt.resid, line='q')
plt.title("Normal QQ plot of the residuals of model_sqrt")
plt.show()


# # Checking normality through homoscedasticity

# In[66]:


plt.scatter(standardized_values(model_sqrt.fittedvalues),
           standardized_values(model_sqrt.resid))
plt.title("Residual Plot of model_sqrt")
plt.xlabel("Standardized Predicted Values")
plt.ylabel("Standardized Residual Values")
plt.show()


# In[67]:


plt.scatter(standardized_values(data2['res_spend']),
           standardized_values(model_sqrt.resid))
plt.title("Residual Plot")
plt.xlabel("Standardized Predicted Values")
plt.ylabel("Standardized Residual Values")
plt.show()


# In[68]:


plt.scatter(standardized_values(data2['admin_spend']),
           standardized_values(model_sqrt.resid))
plt.title("Residual Plot")
plt.xlabel("Standardized Predicted Values")
plt.ylabel("Standardized Residual Values")
plt.show()


# In[69]:


plt.scatter(standardized_values(data2['mkt_spend']),
           standardized_values(model_sqrt.resid))
plt.title("Residual Plot")
plt.xlabel("Standardized Predicted Values")
plt.ylabel("Standardized Residual Values")
plt.show()


# ## Not only normality, variance have drastically improved but also linearity has improved slightly. Hence, we will go ahead with this model

# # COOK'S DISTANCE

# In[72]:


influence = model_sqrt.get_influence()
(c,_) = influence.cooks_distance
c


# In[73]:


# PLot stem plot
plt.figure(figsize=(20,7))
plt.stem(np.arange(len(data2)),np.round(c,3))
plt.title("Plotting influencer values through stem plot")
plt.xlabel("Row Index")
plt.ylabel("Cook's Distance")
plt.show()


# In[74]:


np.argmax(c),np.max(c)


# In[75]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model_sqrt)
plt.show()


# In[77]:


# Calculating leverage cutoff

k=3
n=50
leverage_cutoff=3*((k+1)/n)
leverage_cutoff


# This implies that points 48 and 49 are influence points

# In[81]:


data2=data2.drop(data2.index[[48,49]],axis =0).reset_index()


# # Build model

# In[83]:


model_sqrt.rsquared, model_sqrt.rsquared_adj, model_sqrt.aic


# In[85]:


((model_sqrt.resid)**2).mean()


# In[86]:


table=pd.DataFrame(columns=['res_spend','admin_spend','mkt_spend'])
table


# In[87]:


table['res_spend'] = pd.Series([120000,70000,95000])
table['admin_spend'] = pd.Series([150000,120000,145000])
table['mkt_spend'] = pd.Series([100000,50000,280000])


# In[88]:


table


# In[89]:


Predicted_Profit=model_sqrt.predict(table)


# In[90]:


table['Predicted_Profit']=Predicted_Profit


# In[91]:


table


# In[92]:


table['Predicted_Profit_actual']=(Predicted_Profit)**2


# In[93]:


table


# In[95]:


Model_prediction = (model_sqrt.predict(data2))**2
Model_prediction


# In[96]:


r={"Variables":["Old_rsquared","Final_rsquared"], "Values":[model.rsquared,model_sqrt.rsquared]}
comparison=pd.DataFrame(r)
comparison


# In[ ]:




