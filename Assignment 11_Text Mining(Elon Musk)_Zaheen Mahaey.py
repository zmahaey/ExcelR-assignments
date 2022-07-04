#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Loading the positive opinion Lexicon file
file = pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 11_Text Mining\\positive-words.txt",
                  error_bad_lines=False)
file.head(50)


# In[3]:


# Dropping the 24 rows in data cleansing
file = file.drop(range(0,25),axis=0)
file


# In[4]:


x= file.values
x


# In[5]:


table_pos=pd.DataFrame(x,columns=['words'])
table_pos


# In[6]:


# Assigning all the positive words a value of 1
table_pos['values']=1
table_pos


# In[7]:


# Loading the negative opinion Lexicon file
file2=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 11_Text Mining\\negative-words.txt",
                 error_bad_lines=False,encoding='latin1')
file2.head(50)


# In[8]:


# Dropping the 24 rows in data cleansing
file2=file2.drop(range(0,25),axis=0)
file2


# In[9]:


y=file2.values


# In[10]:


table_neg=pd.DataFrame(y,columns=['words'])
table_neg


# In[11]:


# Assigning all the negative words a value of -1
table_neg['values']=-1
table_neg


# In[12]:


# Appending the positive and negative lexicon dataframes
opinion = table_pos.append(table_neg,ignore_index=True)
opinion


# In[13]:


opinion=opinion.sort_values(['words'])
opinion


# In[14]:


# Loading the Elon Musk tweets file
df=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 11_Text Mining\\Elon_musk.csv",
              encoding='latin1',error_bad_lines=False)
df


# In[15]:


df=df.drop('Unnamed: 0',axis=1)
df.head()


# In[16]:


# Defining the function to clean the data
import re
import string
def clean_data(sentence):
    sentence = re.sub('http[s]?://\S+', '',sentence)# removing the url
    sentence= re.sub('@[\w]+', '',sentence)# removing the twitter handle
    exclude = string.punctuation
    sentence = ''.join([ch for ch in sentence if ch not in exclude])# removing the punctuation
    return sentence


# In[17]:


df['text_new']=df['Text'].apply(clean_data)
df=df.drop('Text', axis=1)
df.head(10)


# In[18]:


# Removing the ascii characters
series=df['text_new'].str.encode('ascii', 'ignore').str.decode('ascii')
series.head()


# In[19]:


type(series)


# In[20]:


df['Tweet']=pd.Series(series)
df.head(5)


# In[21]:


df=df.drop('text_new',axis=1)
df.head(50)


# In[22]:


# Removing the \n character
df['Tweet'] = df['Tweet'].str.replace('\n', '')


# In[23]:


df.head(35)


# In[24]:


type(df)


# In[25]:


import nltk
nltk.download('punkt')
from nltk import tokenize
from nltk.tokenize import word_tokenize
#Remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
import spacy
#!python -m spacy download en_core_web_md


# In[26]:


my_stop_words = stopwords.words('english')
my_stop_words[0:50]


# In[27]:


# defining function to carry out tokenization, stopword removal and lemmatization
import string
nlp = spacy.load('en_core_web_md')

def func(text):
    tokenize_text= word_tokenize(text)
    lower_text = [x.lower() for x in tokenize_text]
    no_stop_words=[word for word in lower_text if word not in my_stop_words]
    tweet=' '.join(no_stop_words)
    sentence= nlp(tweet)
    lemmas = [x.lemma_ for x in sentence]
    final_tweet=' '.join(lemmas)
    return final_tweet


# In[28]:


df['clean']=df['Tweet'].apply(func)
df


# In[29]:


# Converting dataframe to dictionary
affinity_score=opinion.set_index('words')['values'].to_dict()
affinity_score


# In[30]:


# Defining function to calculate sentiment score of the sentence/tweet
lexicons=affinity_score
def sentiment_score(text):
    sent_score=0
    sentence=nlp(text)
    for word in sentence:
        sent_score += lexicons.get(word.lemma_,0)
    return sent_score
        


# In[31]:


df['senti_score']=df['clean'].apply(sentiment_score)


# In[32]:


df


# In[33]:


#Defining function to categorize the tweet as positive or negative or neutral
def tweet_score(num):
    if num > 0:
        return("Positve Tweet")
    elif num == 0:
        return("Neutral Tweet")
    else:
        return("Negative Tweet")


# In[34]:


df['Tweet_sentiment']= df['senti_score'].apply(tweet_score)
df


# In[36]:


df = df.rename({'clean':'Tweet_final'},axis=1)
df.head()


# In[37]:


tweets=df[['Tweet_final','senti_score','Tweet_sentiment']]
tweets.head()


# In[80]:


p_tweets=tweets[tweets['senti_score']>0]
p_tweets


# In[59]:


positive=len(p_tweets)
positive


# In[51]:


neg_tweets=tweets[tweets['senti_score']<0]
neg_tweets


# In[57]:


negative=len(neg_tweets)
negative


# In[53]:


neu_tweets=tweets[tweets['senti_score']==0]
neu_tweets


# In[55]:


neutral=len(neu_tweets)
neutral


# In[69]:


r = {'Tweets_type':['Postive Tweets','Negative Tweets','Neutral Tweets'],'No.of tweets': [positive,negative,neutral]}
d=pd.DataFrame(r)
d


# In[76]:


import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.title("Dataset Sentimental Analysis")
plt.bar(d['Tweets_type'],d['No.of tweets'])


# In[81]:


# Calculating the word count in a tweet
tweets['word_count']= tweets['Tweet_final'].str.split().apply(len)
tweets


# In[84]:


tweets['index']=range(len(tweets))
tweets


# In[85]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
sns.lineplot(y='senti_score',x='index',data=tweets)


# # Generating wordcloud

# In[91]:


text = tweets[['Tweet_final']]
text


# In[92]:


text=[Tweet_final.strip() for Tweet_final in df.Tweet_final]
text=[Tweet_final for Tweet_final in text if Tweet_final]


# In[93]:


text


# In[94]:


final_text=' '.join(text)
final_text


# In[95]:


pip install wordcloud


# In[96]:


# Import packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS


# In[97]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud) 
    plt.axis("off");


# In[99]:


# Generate wordcloud
stopwords = STOPWORDS
stopwords.add('haha')
stopwords.add('rt')
stopwords.add('amp')

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,
                      colormap='Set2',stopwords=stopwords).generate(final_text)
# Plot
plot_cloud(wordcloud)


# In[ ]:




