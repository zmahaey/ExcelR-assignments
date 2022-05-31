#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading data
car=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 5_Multiple regression\\Toyota_Corolla.csv",encoding='latin1')
car.head()


# In[3]:


# Dropping unnecessary columns to obtain the required 9 columns of the dataset
car=car.drop(car.iloc[:,18:38],axis=1)
car=car.drop(car.columns[[0,1,4,5,7,9,10,11,14]],axis=1)
car.head()


# In[4]:


# Renaming the columns
car=car.rename({'Age_08_04': 'Age', 'cc':'CC'},axis=1)
car.head()


# In[5]:


car.shape


# In[6]:


car.info()


# In[7]:


# Checking for null values
car.isnull().sum()


# In[8]:


# Checking for duplicate rows
car[car.duplicated()]


# In[9]:


car[car['KM']==13253]


# In[10]:


#Dropping the duplicate rows
car=car.drop_duplicates()
car.shape


# In[11]:


car[car.duplicated()]


# In[12]:


car.corr()


# In[13]:


sns.set_style(style='darkgrid')
sns.pairplot(car)


# In[15]:


# Defining the regression model
import statsmodels.formula.api as smf
model=smf.ols("Price ~ Age + KM + HP + CC + Doors + Gears + Quarterly_Tax + Weight", data=car).fit()


# In[16]:


model.fittedvalues


# In[17]:


model.params


# In[18]:


model.pvalues, model.tvalues


# In[19]:


model.rsquared,model.rsquared_adj 


# In[20]:


# Calculating VIF (Variance inflation factor)
rsq_Age = smf.ols("Age ~ KM + HP + CC + Doors + Gears + Quarterly_Tax + Weight", data=car).fit().rsquared
vif_Age = 1/(1-rsq_Age)

rsq_KM = smf.ols("KM ~ Age + HP + CC + Doors + Gears + Quarterly_Tax + Weight", data=car).fit().rsquared
vif_KM = 1/(1-rsq_KM)

rsq_HP = smf.ols("HP ~ Age + KM + CC + Doors + Gears + Quarterly_Tax + Weight", data=car).fit().rsquared
vif_HP = 1/(1-rsq_HP)

rsq_CC = smf.ols("CC ~ Age + KM + HP + Doors + Gears + Quarterly_Tax + Weight", data=car).fit().rsquared
vif_CC = 1/(1-rsq_CC)

rsq_Doors = smf.ols("Doors ~ Age + KM + HP + CC + Gears + Quarterly_Tax + Weight", data=car).fit().rsquared
vif_Doors = 1/(1-rsq_Doors)

rsq_Gears = smf.ols("Gears ~ Age + KM + HP + CC + Doors + Quarterly_Tax + Weight", data=car).fit().rsquared
vif_Gears = 1/(1-rsq_Gears)

rsq_Quarterly_Tax = smf.ols("Quarterly_Tax ~ Age + KM + HP + CC + Doors + Gears + Weight", data=car).fit().rsquared
vif_Quarterly_Tax = 1/(1-rsq_Quarterly_Tax)

rsq_Weight = smf.ols("Weight ~ Age + KM + HP + CC + Doors + Gears + Quarterly_Tax", data=car).fit().rsquared
vif_Weight = 1/(1-rsq_Weight)

d1={'Variables': ['Age','KM','HP','CC','Doors','Gears','Quarterly_Tax','Weight'], 'Values':[vif_Age,vif_KM,vif_HP,
                                                                                          vif_CC,vif_Doors,vif_Gears,
                                                                                          vif_Quarterly_Tax,vif_Weight]}
dataframe=pd.DataFrame(d1)
dataframe


# Since VIF < 20 for every variable, the multicollinearity condition does not exist.

# # Residual Analysis

# In[21]:


# QQ Plot
import statsmodels.api as sm
qqplot = sm.qqplot(model.resid, line='q')
plt.title("Normal QQ Plot for residuals")
plt.show()


# In[22]:


sns.boxplot(model.resid)


# In[23]:


sns.histplot(model.resid)


# In[24]:


sns.distplot(model.resid)


# # Checking the normality through Homoscedasticity

# In[25]:


def standardized_values(vals):
    return(vals-vals.mean())/vals.std()


# In[26]:


plt.scatter(standardized_values(model.fittedvalues),
           standardized_values(model.resid))
plt.title("Residual Plot")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# # Residual vs Regressors

# In[27]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"Age", fig= fig)
plt.show()


# In[28]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"KM", fig= fig)
plt.show()


# In[29]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"HP", fig= fig)
plt.show()


# In[30]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"CC", fig= fig)
plt.show()


# In[31]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"Doors", fig= fig)
plt.show()


# In[32]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"Gears", fig= fig)
plt.show()


# In[33]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"Quarterly_Tax", fig= fig)
plt.show()


# In[34]:


fig=plt.figure(figsize= (15,8))
fig=sm.graphics.plot_regress_exog(model,"Weight", fig= fig)
plt.show()


# ### From the pairplot we can infer that linearity could be improved. Similarly, QQ Plot, residual and homoscedasticity analysis tell us that normality, variance could be improved too.Hence, applying square root transformation on "Price"

# # Square root transformation

# In[108]:


cars = car.copy()
car.head()


# In[54]:


sqrt_price = np.sqrt(cars['Price'])


# In[55]:


cars['sqrt_price']=sqrt_price
cars.head()


# In[57]:


# New regression model
model_sqrt=smf.ols("sqrt_price ~ Age + KM + HP + CC + Doors + Gears + Quarterly_Tax + Weight", data=cars).fit()


# In[58]:


model_sqrt.rsquared


# # Residual Analysis

# In[64]:


qqplot = sm.qqplot(model_sqrt.resid, line='q')
plt.title("Normal QQ Plot for residuals- model_sqrt")
plt.show()


# In[61]:


sns.boxplot(model_sqrt.resid)


# In[62]:


sns.histplot(model_sqrt.resid)


# # Checking normality through Homoscedasticity

# In[65]:


plt.scatter(standardized_values(model_sqrt.fittedvalues),
           standardized_values(model_sqrt.resid))
plt.title("Residual Plot")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# In[71]:


plt.scatter(standardized_values(cars['Age']),
           standardized_values(model_sqrt.resid))
plt.title("Residual vs Age")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# In[72]:


plt.scatter(standardized_values(cars['KM']),
           standardized_values(model_sqrt.resid))
plt.title("Residuals vs KM")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# In[73]:


plt.scatter(standardized_values(cars['HP']),
           standardized_values(model_sqrt.resid))
plt.title("Residual vs HP")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# In[74]:


plt.scatter(standardized_values(cars['CC']),
           standardized_values(model_sqrt.resid))
plt.title("Residuals vs CC")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# In[75]:


plt.scatter(standardized_values(cars['Doors']),
           standardized_values(model_sqrt.resid))
plt.title("Residuals vs Doors")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# In[76]:


plt.scatter(standardized_values(cars['Gears']),
           standardized_values(model_sqrt.resid))
plt.title("Residuals vs Gears")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# In[77]:


plt.scatter(standardized_values(cars['Quarterly_Tax']),
           standardized_values(model_sqrt.resid))
plt.title("Residuals vs Quarterly_Tax")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# In[78]:


plt.scatter(standardized_values(cars['Weight']),
           standardized_values(model_sqrt.resid))
plt.title("Residuals vs Weight")
plt.xlabel("Predicted Satndardized Values")
plt.ylabel("Predicted Residual Values")
plt.show


# ### It can be seen from QQplot, residual analysis and plot from residuals vs regressors that the square root transformation has improved the normality and variance

# # Cook's Distance

# In[79]:


# Calculating cook's distance
influence = model_sqrt.get_influence()
(c,_)= influence.cooks_distance
c


# In[80]:


# Plotting stem plot for influencers
plt.figure(figsize = (20,7))
plt.stem(np.arange(len(cars)), np.round(c,3))
plt.xlabel("Row Index")
plt.ylabel("Cook's Distance")
plt.show()


# In[81]:


np.argmax(c),np.max(c)


# In[82]:


from statsmodels.graphics.regressionplots import influence_plot


# In[83]:


# Plotting influence points
influence_plot(model)
plt.show()


# In[84]:


# Calculating Leverage_cutoff

k=8
n=1435
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[88]:


car1=cars.drop(car.index[80],axis=0).reset_index()


# In[89]:


# Plotting stemm plot and influence plot for the remaining data to identify outliers
car1.head()


# In[90]:


model_sqrt1 =smf.ols("sqrt_price ~ Age + KM + HP + CC + Doors + Gears + Quarterly_Tax + Weight", data=car1).fit()


# In[91]:


influence = model1.get_influence()
(c,_)= influence.cooks_distance
c


# In[92]:


plt.figure(figsize = (20,7))
plt.stem(np.arange(len(car1)), np.round(c,3))
plt.xlabel("Row Index")
plt.ylabel("Cook's Distance")
plt.show()


# In[95]:


np.argmax(c),np.max(c)


# In[96]:


influence_plot(model1)
plt.show()


# In[97]:


# Calculating Leverage_cutoff

k=8
n=1434
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[98]:


car2=car1.drop(car.index[[219,599,958,108,109,989]],axis=0).reset_index()


# In[100]:


model_sqrt2=smf.ols("sqrt_price ~ Age + KM + HP + CC + Doors + Gears + Quarterly_Tax + Weight", data=car2).fit()


# In[102]:


model_sqrt2.rsquared, model_sqrt2.rsquared_adj, model_sqrt2.aic


# In[103]:


((model_sqrt2.resid)**2).mean()


# # Model Prediction

# In[104]:


# Creating new data in dataframe

new_data = pd.DataFrame({'Age': 50, 'KM':50000,'HP':98, 'CC':2000, 'Doors':4, 'Gears':5, 'Quarterly_Tax': 210, 
                        'Weight': 1170},index=[0])
new_data


# In[106]:


# Transforming the new data created above by squaring values
price_predicted = (model_sqrt2.predict(new_data))**2
price_predicted


# In[107]:


# Model predicting price values
(model_sqrt2.predict(car2))**2


# In[ ]:




