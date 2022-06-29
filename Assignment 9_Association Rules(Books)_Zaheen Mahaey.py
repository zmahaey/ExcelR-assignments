#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[2]:


# Load the book dataset
book = pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 9_Association Rules\\book.csv")
book


# In[4]:


# Setting the parameters of support and length
freq_itemset= apriori(book,min_support=0.2,use_colnames=True, max_len=2)
freq_itemset


# In[5]:


# Setting the threshold of lift to 0.7
rule=association_rules(freq_itemset, metric='lift',min_threshold=0.7)
rule


# In[6]:


# Plotting the data
import matplotlib.pyplot as plt
plt.plot(rule['support'],rule['confidence'],'r*')
for i,j in zip(rule['support'], round(rule['confidence'],3)):
    plt.text(i,j,'({},{})'.format(i,j))
plt.xlabel("Support value")
plt.ylabel("Confidence value")
plt.title("Plot with minimum support = 0.2")


# ### As lift >1 for both the above points we can infer that both CookBks and childBks can be antecedent and consequent to each other and vice versa.

# In[9]:


# Varying the values of support and length
freq_items= apriori(book,min_support=0.18, use_colnames=True,max_len=2)
freq_items


# In[10]:


rules=association_rules(freq_items,metric='lift',min_threshold=0.7)
rules


# In[11]:


rules.sort_values('confidence', ascending=False)


# In[12]:


# Plotting support and confidence in scatterplot with values as lift
import seaborn as sns
sns.scatterplot(rules['support'],rules['confidence'],hue=round(rules['lift'],2),palette='tab10')


# In[13]:


rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['antecedents']


# In[14]:


rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))


# In[15]:


# Creating pivot table
table=rules.pivot(index='consequents',columns='antecedents',values='support')
table


# In[16]:


# Generting heatmap from pivot
sns.heatmap(table,annot=True,linewidth=1, linecolor='w')# can try cmap= tab10


# In[17]:


rules['rule'] = rules.index
rules['rule']


# In[18]:


coords = rules[['antecedents','consequents','rule']]


# In[19]:


# Plotting parallel coordinates plot
from pandas.plotting import parallel_coordinates
parallel_coordinates(coords,'rule',colormap='tab10')


# In[20]:


# Varying the length and support parameters
freq_item1=apriori(book,min_support = 0.05,use_colnames=True,max_len=5)
freq_item1


# In[21]:


rule1=association_rules(freq_item1,metric='lift',min_threshold=0.7)
rule1


# In[22]:


rule1_filter=rule1[(rule1['support']>0.05) & (rule1['support']<0.18) & (rule1['lift']>3) & (rule1['confidence']>0.5)]
rule1_filter


# # Visualization through scatter plot

# In[23]:


plt.scatter(rule1_filter['antecedent support'], rule1_filter['consequent support'])
plt.xlabel("Antecedent support")
plt.ylabel("Consequent Support")
plt.title("Antecedent support vs Consequent Support")
plt.show()


# In[24]:


plt.scatter(rule1_filter['lift'], rule1_filter['confidence'])
plt.xlabel("Lift")
plt.ylabel("Confidence")
plt.title("lift vs confidence")
plt.show()


# In[25]:


# Lift vs support
plt.scatter(rule1_filter['lift'], rule1_filter['support'])
plt.xlabel("Lift")
plt.ylabel("support")
plt.title("lift vs support")
plt.show()


# In[26]:


rule1_filter['antecedents'] = rule1_filter['antecedents'].apply(lambda a: ','.join(list(a)))
rule1_filter['antecedents']


# In[27]:


rule1_filter['consequents'] = rule1_filter['consequents'].apply(lambda a: ','.join(list(a)))
rule1_filter['consequents']


# In[28]:


# Creating pivot table for the given conditions above
table1 =rule1_filter.pivot(index='antecedents', columns='consequents', values='lift')
table1


# In[29]:


# Generating heatmap for above analysis
sns.heatmap(table1, annot=True, linewidth=1,linecolor='w',cmap='tab10')


# ### Thus we see that for 0.05<support<0.18 , confidence greater than 50% and lift greater than 3, heatmap represents the analysis of what goes with what. For example, If people buy ItalCook book they are quite likely to buy YouthBks and CookBks along with it 

# In[ ]:




