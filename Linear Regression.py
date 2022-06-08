#!/usr/bin/env python
# coding: utf-8

# In[20]:


#Problem Statement
#Build a model which predicts sales based on the money spent on different platforms for marketing.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


df = pd.read_csv('/Users/sbogadhi/Desktop/SALES.csv')


# In[78]:


df.head(10)


# In[69]:


df.shape


# In[16]:


print(df.dtypes)


# In[17]:


df.isnull().any()


# In[18]:


df.info()


# In[79]:


df.describe()


# In[28]:


dataset.plot(x='TV', y='sales', style='o')


# In[29]:


dataset.plot(x='radio', y='sales', style='o')


# In[30]:


dataset.plot(x='newspaper', y='sales', style='o')


# In[41]:


dataset.corr()


# In[43]:


import seaborn as sns


# In[42]:


sns.heatmap(dataset.corr())


# In[45]:


sns.heatmap(dataset.corr(), annot=True)


# In[50]:


X = dataset['TV'].values.reshape(-1,1)
y = dataset['sales'].values.reshape(-1,1)


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[74]:


regressor = LinearRegression()


# In[53]:


regressor.fit(X_train, y_train) #training the algorithm


# In[54]:


#To retrieve the intercept:
print(regressor.intercept_)


# In[55]:


#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)


# In[56]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[57]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[65]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[66]:


### R-squared ###
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[72]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 7.29+0.04*X_test, 'r')
plt.show()


# In[ ]:




