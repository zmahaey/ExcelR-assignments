#!/usr/bin/env python
# coding: utf-8

# In[109]:


import pandas as pd
import numpy as np


# In[110]:


# Load the data
book=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 10_Recommendation system\\book.csv",
                encoding='latin1')
book


# In[111]:


#Dropping unnecessary column
book = book.drop('Unnamed: 0',axis=1)
book


# In[112]:


book.info()


# In[113]:


book.isnull().sum()


# In[114]:


len(book['User.ID'].unique())


# In[115]:


len(book['Book.Title'].unique())


# In[116]:


# Getting rating values
book['Book.Rating'].unique()


# In[117]:


# Identifying duplicate entries
book[book.duplicated()]


# In[89]:


book[book['Book.Title']=="The Magician's Tale" ] 


# In[90]:


book[book['Book.Title']=="Le nouveau soleil de Teur" ] 


# In[91]:


# Dropping duplicates
book=book.drop_duplicates()
book.shape


# In[92]:


book1= book.reset_index()
book1.head()


# In[138]:


# Sorting columns by userdID and book title
book1.sort_values(['User.ID', 'Book.Title'])


# In[140]:


# Creating pivot table
user_books_df = book1.pivot_table(index='User.ID',
                 columns='Book.Title',
                 values='Book.Rating').fillna(0)

user_books_df


# In[141]:


# Calculating cosine similarity
user_sim = 1 - pairwise_distances(user_books_df.values, metric= 'cosine')


# In[142]:


user_sim


# In[143]:


# Creating dataframe for user similarity
user_sim_df = pd.DataFrame(user_sim)


# In[144]:


user_sim_df


# In[145]:


# Assigning User ID to index and column
user_sim_df.index = list(user_books_df.index)
user_sim_df.columns = list(user_books_df.index)
user_sim_df


# In[146]:


# Assigning diagonal values to be zero
np.fill_diagonal(user_sim, 0)
user_sim_df


# In[147]:


# User similarity analysis
user_sim_df.idxmax(axis=1)[0:10]


# In[153]:


book[(book['User.ID']==19) | (book['User.ID']==278418)]


# In[154]:


book[book['Book.Title']=='The Murder Book']


# In[149]:


book[(book['User.ID']==53) | (book['User.ID']==1996)]


# In[ ]:




