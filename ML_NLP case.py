#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np   
import pandas as pd  
  
# Import dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')  


# In[14]:


dataset.shape


# In[5]:


dataset.head(15)


# In[4]:


# library to clean data 
import re  
  
# Natural Language Tool Kit 
import nltk  
  
nltk.download('stopwords') 
  
# to remove stopword 
from nltk.corpus import stopwords 
  
# for Stemming propose  
from nltk.stem.porter import PorterStemmer 
  


# In[53]:


print(stopwords.words('english'))


# In[54]:


# Initialize empty array 
# to append clean text  
corpus = []  
  
# 1000 (reviews) rows to clean 
for i in range(0, 1000):  
      
    # column : "Review", row ith 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  
      
    # convert all cases to lower cases 
    review = review.lower()  
      
    # split to array(default delimiter is " ") 
    review = review.split()  
      
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = PorterStemmer()  
      
    # loop for stemming each word 
    # in string array at ith row     
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]  
    # rejoin all string array elements 
    # to create back into a string 
    review = ' '.join(review)   
      
    # append each string to create 
    # array of clean text  
    corpus.append(review)  


# In[32]:


df1=dataset[6:7]


# In[40]:


df1['Review'][6]


# In[41]:


df1


# In[42]:


review = re.sub('[^a-zA-Z]', ' ', df1['Review'][6])  


# In[43]:


review


# In[49]:


review1 =  review.lower() 
review1


# In[50]:


review2 = review1.split() 
review2


# In[52]:


ps = PorterStemmer()
review3 = [ps.stem(word) for word in review2 
                if not word in set(stopwords.words('english'))]
review3


# In[55]:


# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 
  
# To extract max 1500 feature. 
# "max_features" is attribute to 
# experiment with to get better results 
cv = CountVectorizer(max_features = 1500)  
  
# X contains corpus (dependent variable) 
X = cv.fit_transform(corpus).toarray()  
  
# y contains answers if review 
# is positive or negative 
y = dataset.iloc[:, 1].values  


# In[61]:


corpus


# In[68]:


corpus1=corpus[0:4]
corpus1


# In[71]:


cv1= CountVectorizer()


# In[72]:


X1 = cv1.fit_transform(corpus1)


# In[73]:


print(cv1.get_feature_names())


# In[74]:


print(X1.toarray())


# In[80]:


# Splitting the dataset into 
# the Training set and Test set 
from sklearn.model_selection import train_test_split 
  
# experiment with "test_size" 
# to get better results 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 


# In[81]:


# Fitting Random Forest Classification 
# to the Training set 
from sklearn.ensemble import RandomForestClassifier  
  
# n_estimators can be said as number of 
# trees, experiment with n_estimators 
# to get better results  
model = RandomForestClassifier(n_estimators = 501, 
                            criterion = 'entropy') 
                              
model.fit(X_train, y_train)  


# In[82]:


y_pred = model.predict(X_test) 
  
y_pred 


# In[83]:


from sklearn.metrics import confusion_matrix 
  
cm = confusion_matrix(y_test, y_pred) 
  
cm 


# In[87]:


(16+47)/250


# In[88]:


(109+78)/250


# In[ ]:




