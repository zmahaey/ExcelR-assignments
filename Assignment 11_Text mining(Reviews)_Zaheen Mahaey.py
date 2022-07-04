#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[91]:


# Loading reviews file created from Amazon website
file= pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 11_Text Mining\\reviews.csv",
                 encoding='latin1')
file


# In[92]:


# Defining the function to clean the data
import re
def clean(sentence):
    sentence=re.sub(r'\s+|\\n', ' ',sentence)# removing \n
    return sentence

file['text']= file['review'].apply(clean)
file


# In[93]:


df=file[['text']]
df


# In[94]:


import nltk
nltk.download('punkt')


# In[95]:


#Remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
import spacy


# In[96]:


my_stop_words = stopwords.words('english')
my_stop_words[0:10]


# In[97]:


# Defining the function to tokenize,remove stopwords and lemmatize the sentence
import string
from nltk.tokenize import word_tokenize
nlp = spacy.load('en_core_web_md')

def func(text):
    tokenize_text= word_tokenize(text)
    lower_text = [x.lower() for x in tokenize_text]
    no_stop_words=[word for word in lower_text if word not in my_stop_words]
    text=' '.join(no_stop_words)
    sentence= nlp(text)
    lemmas = [x.lemma_ for x in sentence]
    final_text=' '.join(lemmas)
    return final_text


# In[98]:


df['lemmas']=df['text'].apply(func)
df


# In[100]:


# Loading the affinity lexicon file
affin=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\NLP\\Text mining\\Afinn.csv", encoding='latin1')
affin.head()


# In[103]:


affinity_score= affin.set_index('word')['value'].to_dict()
affinity_score


# In[104]:


#Defining the function to claculate the sentiment score 
import spacy
nlp = spacy.load('en_core_web_md')
sentiment_lexicon = affinity_score

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)# If lemmatized word doen't exist return 0
    return sent_score


# In[105]:


df['sentiment_value'] = df['lemmas'].apply(calculate_sentiment)
df


# In[107]:


df_final = df[['text','sentiment_value']]
df_final


# In[109]:


# how many words are in the sentence?
df_final['word_count'] = df_final['text'].str.split().apply(len)
df_final


# In[111]:


df_final['index']=range(len(df_final))
df_final


# In[112]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
sns.lineplot(y='sentiment_value',x='index',data=df_final)


# # Generating wordcloud for laptop reviews

# In[141]:


t=df[['text']]
t


# In[142]:


t=[text.strip() for text in t.text]
t=[text for text in t if text]
t


# In[147]:


final_t=' '.join(t)
final_t


# In[148]:


# Import packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS


# In[149]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud) 
    plt.axis("off");


# In[151]:


# Generate wordcloud
stopwords = STOPWORDS

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=75,
                      colormap='Set2',stopwords=stopwords).generate(final_t)
# Plot
plot_cloud(wordcloud)


# In[ ]:




