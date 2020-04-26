
# coding: utf-8

# In[1]:

import os
import warnings  # to filter warnings which is quite annoying to look at.
# To perform various mathematical operations and tools to operate on nd arrays
import numpy as np
import pandas as pd  # To import and analyze data
#import matplotlib.pyplot as plt  # for visualisation
#import seaborn as sns  # for visualisation
import pickle  # To save data or python objects from primary memory to disk and store it in a binary format vice versa
# Natural language processing toolkit used for vectorisation and preprocesssing data.
#import nltk
#get_ipython().magic('matplotlib inline')
# Output of plotting commands to be displayed inline in the jupyter notebook.
warnings.filterwarnings('ignore')


# In[2]:

dataset = pd.read_csv("cropsss.csv")
dataset = dataset[1:]


# In[3]:

dataset.head(15)


# In[4]:

## Seperating the attributes and class label


X=dataset.drop(["Crop"],axis=1)
y=dataset["Crop"]


# In[5]:

dataset.dtypes


# In[6]:

y.value_counts()


# In[7]:

## Train Test Split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle="False") 
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[8]:

### Random Forests classifier



from sklearn.ensemble import RandomForestClassifier 

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

a=accuracy_score(y_test, pred)
print(a)

x = [[2.9,0.35,2.8,47,250,15.1,301,55.5,7,6.55,54,8]]
y=classifier.predict(x)
print(y)


# In[9]:

### Calculating accuracy


from sklearn.metrics import accuracy_score

a=accuracy_score(y_test, pred)
print(a)


# In[20]:

#import scikitplot as splotst))


# In[53]:

## Predicting classes for discrete instances
x = [[2.9,0.35,2.8,47,250,15.1,301,55.5,7,6.55,54]]
y=classifier.predict(x)
print(y)


# In[56]:

## Predicting classes for discrete instances
x = [[2,0.35,1,47,250,15.1,55,125,6.5,6.5,25]]
y=classifier.predict(x)
print(y)


# In[59]:

## Predicting classes for discrete instances
x = [[4,0.35,1.5,47,245,15.1,55,125,6.5,5,36]]
y=classifier.predict(x)
print(y)


# In[61]:

## Predicting classes for discrete instances
x = [[5,0.35,1.5,47,245,15.1,55,125,6.5,5,50]]
y=classifier.predict(x)
print(y)


# In[62]:



### Decision Trees doesnt provide good results

from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
from sklearn.metrics import accuracy_score

a=accuracy_score(y_test, dtree_predictions)
print(a)
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, dtree_predictions) 


# In[ ]:



