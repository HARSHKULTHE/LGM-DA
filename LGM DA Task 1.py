#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("C:\\Users\\Harsh\\Desktop\\lgm da\\iris.csv")


# In[3]:


data.head()


# In[4]:


data.sample(10)


# In[5]:


data.columns


# In[6]:


data.shape


# In[7]:


print(data)


# In[8]:


#data[start:end]
#start is inclusive whereas end is exclusive
print(data[10:21])
# it will print the rows from 10 to 20.

# you can also save it in a variable for further use in analysis
sliced_data=data[10:21]
print(sliced_data)


# In[9]:


#here in the case of Iris dataset
#we will save it in a another variable named "specific_data"

specific_data=data[["id","class"]]
#data[["column_name1","column_name2","column_name3"]]

#now we will print the first 10 columns of the specific_data dataframe.
print(specific_data.head(10))


# In[10]:


data.iloc[5]
data.loc[data["class"] == "Iris-setosa"]


# In[11]:


data["class"].value_counts()


# In[12]:


# data["column_name"].sum()

sum_data = data["sepallength"].sum()
mean_data = data["sepallength"].mean()
median_data = data["sepallength"].median()

print("Sum:",sum_data, "\nMean:", mean_data, "\nMedian:",median_data)


# In[13]:


min_data=data["sepallength"].min()
max_data=data["sepallength"].max()

print("Minimum:",min_data, "\nMaximum:", max_data)


# In[14]:


# For example, if we want to add a column let say "total_values",
# that means if you want to add all the integer value of that particular
# row and get total answer in the new column "total_values".
# first we will extract the columns which have integer values.
cols = data.columns

# it will print the list of column names.
print(cols)

# we will take that columns which have integer values.
cols = cols[1:5]

# we will save it in the new dataframe variable
data1 = data[cols]

# now adding new column "total_values" to dataframe data.
data["total_values"]=data1[cols].sum(axis=1)

# here axis=1 means you are working in rows,
# whereas axis=0 means you are working in columns.


# In[15]:


newcols={
"id":"Id",
"sepallength":"SepalLengthCm",
"sepalwidth":"SepalWidthCm"}

data.rename(columns=newcols,inplace=True)

print(data.head())


# In[16]:


data.style


# In[17]:


data.isnull()
#if there is data is missing, it will display True else False.


# In[18]:


data.describe()


# In[19]:


data.info()


# In[22]:


plt.figure(figsize = (10, 7))
x = data["SepalLengthCm"]

plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")


# In[24]:


plt.figure(figsize = (10, 7))
x = data.SepalWidthCm

plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.show()


# In[25]:


plt.figure(figsize = (10, 7))
x = data.petallength

plt.hist(x, bins = 20, color = "green")
plt.title("Petal Length in cm")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Count")
plt.show()


# In[26]:


plt.figure(figsize = (10, 7))
x = data.petalwidth

plt.hist(x, bins = 20, color = "green")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")

plt.show()


# In[28]:


# removing Id column
new_data = data[["SepalLengthCm", "SepalWidthCm", "petallength", "petalwidth"]]
print(new_data.head())


# In[29]:


plt.figure(figsize = (10, 7))
new_data.boxplot()


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
iris = pd.read_csv("C:\\Users\\Harsh\\Desktop\\lgm da\\Iris.csv")

plt.plot(iris.id, iris["sepallength"], "r--")
plt.show


# In[31]:


iris.plot(kind ="scatter",
		x ='sepallength',
		y ='petallength')
plt.grid()


# In[36]:


import seaborn as sns
g = sns.pairplot(data,hue="class")


# In[ ]:




