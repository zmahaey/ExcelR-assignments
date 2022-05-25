#!/usr/bin/env python
# coding: utf-8

# # Hypothesis Testing - Assignment 3 (Zaheen Mahaey)

# ## Answer 1  - Cutlets.csv

# In[1]:


import pandas as pd
import scipy
from scipy import stats


# In[2]:


cutlets = pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 3\\Cutlets.csv")
cutlets


# Null Hypothesis: There is no significant difference in the diameter of the cutlet between two units.
# 
# Alternate Hypothesis: There is significant difference in the diameter of the cutlet between two units.
# 
# We will use two sample T-test

# In[3]:


p=stats.ttest_ind(cutlets['Unit A'], cutlets['Unit B'])


# In[4]:


p


# Since p>0.05, we fail to reject the null hypothesis. Hence we csn say that there is no significant
# difference in the diameter of the cutlet between two units.

# ## Answer 2  - LabTAT.csv

# In[5]:


import pandas as pd
import scipy
from scipy import stats


# In[7]:


lab= pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 3\\LabTAT.csv")
lab


# Null Hypothesis: There is no significant difference in average TAT among the different laboratories.
#     
# Alternate Hypothesis: At least one laboratory has greater TAT among the four given laboratories.
#     
# We use ANOVA here

# In[8]:


d=stats.f_oneway(lab.iloc[:,0],lab.iloc[:,1],lab.iloc[:,2],lab.iloc[:,3])
d


# Since p>0.05, we fail to reject the null hypothesis. Hence, we can say that there is no significant difference
# in average TAT among the different laboratories. 

# ## Answer 3 - Buyer ratio.csv

# In[9]:


import pandas as pd
import scipy
from scipy import stats 


# In[10]:


sales = pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 3\\BuyerRatio.csv")
sales


# Null Hypothesis:Male-female buyer rations are similar across regions
#     
# Alternate Hypothesis: Male-female buyer rations are different across regions

# In[14]:


chisq = pd.DataFrame([[50,142,131,70], [435,1523,1356,750]],
                    index=['Males','Females'],
                    columns = ['East','West','North','South'])
chisq


# In[15]:


stats.chi2_contingency(chisq)


# In[16]:


stats.chi2_contingency(chisq)[1]


# Since p>0.05, we fail to reject the null hypothesis. Hence, we can say that Male-female buyer rations
# are similar across regions

# ## Answer 4 - CustomerOrderForm.csv

# In[17]:


import pandas as pd
import scipy
from scipy import stats


# In[19]:


data=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 3\\CustomerOrderForm.csv")
data


# In[23]:


data['Phillippines'].replace(['Error Free','Defective'],[1,0], inplace=True)
data['Indonesia'].replace(['Error Free','Defective'],[1,0], inplace=True)
data['Malta'].replace(['Error Free','Defective'],[1,0], inplace=True)
data['India'].replace(['Error Free','Defective'],[1,0], inplace=True)
data


# Null Hypothesis: The defective percentage of the order form does not vary by centre
#     
# Alternate Hypothesis: The defective percentage of the order form varies by centre
# 
# We will use ANOVA here

# In[24]:


stats.f_oneway(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2], data.iloc[:,3])


# Since p>0.05, we fail to reject the null hypothesis. Hence, we can say that the defective percentage of the order form
# does not vary by centre.

# In[ ]:




