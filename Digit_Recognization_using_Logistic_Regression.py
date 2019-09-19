
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


from sklearn.datasets import load_digits                       # This contain the dataset of digits


# In[6]:


df = load_digits()
dir(df)                                                       # This will give the various attribute in digit dataset


# In[9]:


df.data[1]                                                    # Image representation in the form of array


# In[12]:


plt.gray()
plt.matshow(df.images[2])                                     # just to see the image at position 2


# In[15]:


from sklearn.model_selection import train_test_split                     # It will split the dataset into test and train


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(df.data,df.target,test_size = 0.2)     # Divide test/train in 20:80 ratio


# In[18]:


len(X_train)              


# In[19]:


len(X_test)


# In[20]:


reg = linear_model.LogisticRegression()                                   # Make the object of Logistic Regression
reg.fit(X_train,y_train)                                                  # Apply logistic regression


# In[21]:


reg.score(X_test,y_test)                                                 # finding the percentage accuracy


# In[23]:


reg.predict([df.data[3]])


# In[26]:


y_prediction = reg.predict(X_test)                                       # Save all the prediction in one data frame
from sklearn.metrics import confusion_matrix                             # import confusion matrix from sklearn
cm = confusion_matrix(y_test,y_prediction)                               # create a confusion matrix b/w actual test result and predicted test result
cm


# In[28]:


import seaborn as sb                                                    # Import seaborn for the visualization of confusion matrix
plt.figure(figsize=(10,7))
sb.heatmap(cm,annot=True)
plt.xlabel('prediction')
plt.ylabel('actual')

