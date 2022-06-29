#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[3]:


#Loading movie dataset
movie=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 9_Association Rules\\my_movies.csv")
movie


# In[4]:


movie=movie.drop(['V1','V2','V3','V4','V5'],axis=1)
movie


# In[41]:


# for min_support = 0.1 and maximum length = 2
freq_itemset = apriori(movie, min_support=0.1,use_colnames=True,max_len=2)
freq_itemset


# In[49]:


# Setting confidence threshold =0.5
rules=association_rules(freq_itemset,metric='confidence',min_threshold=0.5)
rules


# In[50]:


# Displaying results for lift>2
rules=rules[rules['lift']>2]
rules


# In[51]:


# Removing the brackets in the values of antecedents
rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['antecedents']


# In[52]:


rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))
rules['consequents']


# In[57]:


rules['rule']=range(len(rules['antecedents']))
rules['rule']


# In[58]:


coords = rules[['antecedents','consequents','rule']]
coords


# In[59]:


# Plotting Parallel cordinates plot
from pandas.plotting import parallel_coordinates
plot = parallel_coordinates(coords,'rule',colormap='tab10')


# ### Under the given conditions (minimum support = 0.1, confidence= 50% and lift >2) , we can see from the parallel coordinates plot that:
# ### 1) Users who watched movie LOTR1 have a high probability that they will watch Harry Potter1 too & vice versa.
# ### 2) Users who watched movie Green Mile have a high probability that they will watch LOTR & vice versa.

# In[61]:


# Changing the value of maximum length and minimum support 
freq_item = apriori(movie, min_support = 0.05,use_colnames=True,max_len=3)
freq_item


# In[64]:


rule=association_rules(freq_item,metric='confidence',min_threshold=1)
rule


# In[65]:


# Converting consequents into strings
rule['consequents'] = rule['consequents'].apply(lambda a: ','.join(list(a)))
rule['consequents']


# In[66]:


# Converting antecedents into strings
rule['antecedents'] = rule['antecedents'].apply(lambda a: ','.join(list(a)))
rule['antecedents']


# In[69]:


# Creating a pivot table with values of lift
table = rule.pivot(index='antecedents',columns='consequents',values='lift')
table


# In[72]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
sns.heatmap(table,annot=True,cmap='tab10',linewidth=1,linecolor='w')


# ### From the heatmap we can deduce that, for example:
# ### 1) Users who saw movies Gladiator and Green Mile have very high probability of watching LOTR and vice versa.

# # Visualizing rules through scatterplot

# In[73]:


# Plotting support vs confidence
plt.scatter(rule['support'],rule['confidence'])
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")
plt.show()


# In[74]:


# Plotting lift vs confidence
plt.scatter(rule['support'],rule['lift'])
plt.xlabel("Support")
plt.ylabel("Lift")
plt.title("Support vs Lift")
plt.show()


# In[75]:


# Plotting lift vs confidence
plt.scatter(rule['lift'],rule['confidence'])
plt.xlabel("lift")
plt.ylabel("Confidence")
plt.title("lift vs Confidence")
plt.show()


# In[ ]:




