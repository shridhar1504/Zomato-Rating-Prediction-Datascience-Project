#!/usr/bin/env python
# coding: utf-8

# # Zomato Rating Prediction

# ***
# _**Importing the required libraries & packages**_
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import matplotlib.ticker as mtick
plt.style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# _**Changing The Default Working Directory Path & Reading the Dataset using Pandas Command**_

# In[2]:


os.chdir('C:\\Users\\Shridhar\\OneDrive\\Desktop\\Top Mentor\\Batch 74 Day 37\\Project 11 Flask Project and Deployment')
df=pd.read_csv('zomato.csv')


# ## Data Cleaning:
# _**Checking the Null values of all the columns in the dataset.**_

# In[3]:


df.isna().sum()


# _**Dropping the unwanted columns from the dataset**_

# In[4]:


df.drop(['url','phone'],inplace=True,axis=1)


# _**Checking for the duplicate values**_

# In[5]:


df.duplicated().sum()


# _**Dropping all the duplicate values from the dataset**_

# In[6]:


df.drop_duplicates(inplace=True)


# _**Checking for the duplicate values after dropping it**_

# In[7]:


df.duplicated().sum()


# _**Dropping all the null values from the dataset**_

# In[8]:


df.dropna(how='any',inplace=True)


# _**Checking the Null values of all the columns in the dataset after dropping all the null values.**_

# In[9]:


df.isna().sum()


# _**Getting all the column names from the dataset**_

# In[10]:


df.columns


# _**Renaming the columns of the dataset for making it simple**_

# In[11]:


df=df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})


# _**Checking the column names from the dataset after renaming it**_

# In[12]:


df.columns


# _**Getting all the unique values of cost column from the dataset.**_

# In[13]:


df.cost.unique()


# _**Updating the cost column with the appropriate form of numerics**_

# In[14]:


df['cost']=df['cost'].apply(lambda x:x.replace(',',''))


# _**Getting all the unique values of cost column from the dataset after changing it to numeric form**_

# In[15]:


df.cost.unique()


# _**Changing the data type of the cost column from "object" to "float"**_

# In[16]:


df['cost']=df['cost'].astype('float')


# _**Getting the unique value and its counts in the cost Column from the dataset**_

# In[17]:


df.cost.value_counts()


# _**Getting all the unique values of rate column from the dataset.**_

# In[18]:


df.rate.unique()


# _**Updating the rate column without having "NEW" in it for the future processing**_

# In[19]:


df=df[df['rate']!='NEW']


# _**Getting all the unique values of rate column from the dataset after updating the rate column.**_

# In[20]:


df.rate.unique()


# _**Updating the rate column as it can be transformed into proper form of numerics.**_

# In[21]:


df['rate']=df['rate'].apply(lambda x:x.replace('/5',''))


# _**Getting all the unique values of rate column from the dataset after updating it.**_

# In[22]:


df.rate.unique()


# ## Visualizations
# 
# _**Assigning the Top 20 restaurants to the new variable for the simpler visualization**_

# In[23]:


chains=df['name'].value_counts().head(20)


# _**Plotting the bar graph for the most famous restaurants(Top 20 restaurants) and saving the PNG file**_

# In[24]:


plt.figure(figsize=(20,15))
sns.barplot(x=chains,y=chains.index,palette='deep')
plt.title("Most famous restaurants chains in Bengaluru")
plt.xlabel("Number of outlets")
plt.savefig('Most famous restaurants in Bengaluru.png')
plt.show()


# _**Plotting the Pie-Chart with Table Booking Values**_

# In[25]:


x=df['book_table'].value_counts()
colors = ['#800080', '#0000A0']
trace=go.Pie(labels=x.index,values=x,textinfo="value",
            marker=dict(colors=colors, 
                           line=dict(color='#001000', width=3)))
layout=go.Layout(title="Table booking",width=400,height=400)
fig=go.Figure(data=[trace],layout=layout)
py.iplot(fig, filename='pie_chart_subplots')
plt.show()


# _**Plotting The  Bar Plot with the Online Delivery data using Seaborn Count Plot Package and saving the PNG File**_

# In[26]:


sns.countplot(x = 'online_order', data = df)
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Whether Restaurants deliver online or Not')
plt.savefig('Online Delivery of Restaurants.png')
plt.show()


# _**Visualizing the data distribution of the rate column against the density distribution using Seaborn Distplot and saving the PNG file**_

# In[27]:


sns.distplot(df['rate'],bins=20)
plt.title('Data Distribution of Rate Column')
plt.savefig('Data Distribution of Rate Column.png')
plt.show()


# _**Getting the Minimum and Maximum Rating of the Restaurants**_

# In[28]:


display(df['rate'].min())
display(df['rate'].max())


# _**Getting all the unique values of rate column from the dataset.**_

# In[29]:


df.rate.unique()


# _**Changing the data type of the rate column from "object" to "float"**_

# In[30]:


df['rate']=df['rate'].astype('float')


# _**Getting all the unique values of rate column from the dataset after updating the data types.**_

# In[31]:


df.rate.unique()


# _**Plotting the Bar Graph using Matplotlib package with rate column grouping it by an unit difference and saving the PNG file**_

# In[32]:


group= [1,2,3,4,5]
plt.hist (df['rate'],group,histtype = 'bar',rwidth =0.5,color = 'b')
plt.title('Rating Unit Difference size')
plt.savefig('Rating Unit Difference size.png')
plt.show()


# _**Plotting the Bar Graph using Matplotlib package with rate column grouping it by 0.5 unit difference and saving the PNG file**_

# In[33]:


group= [1.5,2,2.5,3,3.5,4,4.5,5]
plt.hist (df['rate'],group,histtype = 'bar',rwidth =0.5,color = 'b')
plt.title('Rating 0.5 Unit Difference size')
plt.savefig('Rating 0.5  Unit Difference size.png')
plt.show()


# _**Assigning the different variables for the rate column for using PieChart to visualize the percentage of restaurants according to ratings**_

# In[34]:


gr_1to2=((df['rate']>=1) & (df['rate']<2)).sum()
gr_2to3=((df['rate']>=2)&(df['rate']<3)).sum()
gr_3to4=((df['rate']>=3) & (df['rate']<4)).sum()
gr_4to5=((df['rate']>4)).sum()


# _**Plotting the Pie Chart with assigned variable to visualize the percentage of restaurants according to ratings and saving the PNG file**_ 

# In[35]:


slices=[gr_1to2,gr_2to3,gr_3to4,gr_4to5]   
labels=['Rating 1 to 2','Rating 2 to 3','Rating 3 to 4','Rating >4']
colors = ['#ff3333','#c2c2d6','#6699ff']
plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2,shadow=True)
fig = plt.gcf()
plt.title("Percentage of Restaurants according to their ratings")

fig.set_size_inches(5,5)
plt.savefig('Percentage of Restaurants according to their ratings.png')
plt.show()


# _**Plotting The Bar Plot with the Service Type data using Seaborn Count Plot Package and saving the PNG File**_

# In[36]:


sns.countplot(x= 'type', data = df).set_xticklabels(sns.countplot(x= 'type' , data = df).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(5,3)
plt.title('Type of Service')
plt.savefig('Type of Service.png')
plt.show()


# _**Grouping by Cost Column from the dataset by their sizes**_

# In[37]:


df.groupby('cost').size()


# _**Plotting the Box Plot with cost values**_

# In[38]:


from plotly.offline import iplot
trace0=go.Box(y=df['cost'],name="accepting online orders",
              marker = dict(
        color = 'rgb(113, 10, 100)',
    ))
data=[trace0]
layout=go.Layout(title="Box plot of approximate cost",width=800,height=800,yaxis=dict(title="Price"))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# _**Visualizing the data distribution of the cost column against the density distribution using Seaborn Distplot and saving the PNG file**_

# In[39]:


plt.figure(figsize=(8,8))
sns.distplot(df['cost'])
plt.title('Data Distribution of Cost Column')
plt.savefig('Data Distribution of Cost Column.png')
plt.show()


# _**By using Regular Expression Package, splitting occurs in dish liked columns as multiple values seperated by comma, Extract each dishes and creating a list by apppending the each dishes**_

# In[40]:


import re
likes=[]
df.index=range(df.shape[0])
for i in range(df.shape[0]):
        array_split=re.split(',',df['dish_liked'][i])
        for item in array_split:
            likes.append(item)


# _**Displaying the Each Number of dishes, its counts and the list of each dishes**_

# In[41]:


display ('Number of dishes', len(likes))
display(likes)


# _**Displaying the indices of dataset**_

# In[42]:


df.index=range(df.shape[0])
display (df.index)


# _**Finding out the most Liked Dishes and getting its value counts and displaying the Top 30 Most liked foods**_

# In[43]:


print("Count of Most liked dishes")
favourite_food = pd.Series(likes).value_counts()
display(favourite_food.head(30))


# _**Plotting the Bar Graph with the Most Liked Foods for the Top 30 foods and Number of likes it got and saving the PNG file**_

# In[44]:


ax = favourite_food.nlargest(n=20, keep='first').plot(kind='bar',figsize=(20,15),title = 'Top 30 Favourite Food counts ')
for i in ax.patches:
    ax.annotate(str(i.get_height()), (i.get_x() * 1.005, i.get_height() * 1.005))
plt.savefig('Top 30 Favourite Food counts.png')
plt.show()


# _**Displaying the value counts of Restaurants Type**_

# In[45]:


print (df['rest_type'].value_counts().head(50))


# _**Plotting the bar graph for the Restaurant types and saving the PNG file**_

# In[46]:


plt.figure(figsize=(15,12))
rest=df['rest_type'].value_counts()[:20]
sns.barplot(x= rest,y =rest.index)
plt.title("Restaurant types")
plt.xlabel("count")
plt.savefig('Restaurant types.png')
plt.show()


# ## Data Preprocessing:
# _**Label Encoding the online order column from the dataset for fitting the algorithms**_

# In[47]:


df['online_order']=df['online_order'].astype('category')
df['online_order']=df['online_order'].cat.codes


# _**Label Encoding the book table column from the dataset for fitting the algorithms**_

# In[48]:


df.book_table[df.book_table=='Yes']=1
df.book_table[df.book_table=='No']=0
df.book_table=pd.to_numeric(df.book_table)


# _**Label Encoding all the required columns from the dataset for fitting the algorithms**_

# In[49]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.location=le.fit_transform(df.location)
df.rest_type=le.fit_transform(df.rest_type)
df.cuisines=le.fit_transform(df.cuisines)
df.menu_item=le.fit_transform(df.menu_item)


# _**Getting all the required columns to the new dataframe and exporting it as a CSV file**_

# In[50]:


new_df=df.iloc[:,[2,3,4,5,6,7,9,10,12]]
new_df.to_csv('Cleaned Data.csv',index=False)


# _**Assigning the dependent and independent variable**_

# In[51]:


x=df.iloc[:,[2,3,5,6,7,9,10,12]]
y=df['rate']


# ## Model Fitting:

# _**Splitting the dependent variable & independent variable into training and test dataset using train test split**_

# In[52]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)


# _**Fitting the Linear Regression model with the train dependent and train independent variable and getting the r2 Score between the predicted value and dependent test dataset.**_

# In[53]:


lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# _**Fitting the Random Forest Regressor model with the train dependent and train independent variable and getting the r2 Score between the predicted value and dependent test dataset**_

# In[54]:


from sklearn.ensemble import RandomForestRegressor
RF_Model=RandomForestRegressor(n_estimators=650,random_state=245,min_samples_leaf=.0001)
RF_Model.fit(x_train,y_train)
y_predict=RF_Model.predict(x_test)
display (r2_score(y_test,y_predict))


# _**Passing some of the list of parameters for the Random Forest Regressor Model to run with Randomized Search CV Algorithm**_

# In[55]:


params={
    "n_estimators":[100,200,300,400,500,600,700,800],
    "max_features":['auto','sqrt'],
    "max_depth":[int(x) for x in np.linspace(5,30,num=6)],
    "min_samples_leaf":[1,2,5,10],
    "min_samples_split":[2,5,10,15,100],
    "min_weight_fraction_leaf":[0.0,0.1,0.2,0.3,0.4,0.5]
}


# _**Fitting The Random Forest Regressor Model with the above mentioned parameters in the RandomizedSearchCV Algorithm**_

# In[56]:


from sklearn.model_selection import RandomizedSearchCV
RF=RandomForestRegressor()
random_search=RandomizedSearchCV(RF,params,n_iter=100,scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)
random_search.fit(x_train,y_train)


# _**Displaying the Best Parameters of the Random Forest Regressor Model**_

# In[57]:


random_search.best_params_


# _**Displaying the Best Estimators of the Random Forest Regressor Model**_

# In[58]:


random_search.best_estimator_


# _**Fitting the Extra Tree Regressor model with the train dependent and train independent variable and getting the r2 Score between the predicted value and dependent test dataset**_

# In[59]:


from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120)
ET_Model.fit(x_train,y_train)
y_predict=ET_Model.predict(x_test)
r2_score(y_test,y_predict)


# _**Create the pickle file of the model with the highest r2 score with the model name**_

# In[60]:


import pickle 
pickle.dump(ET_Model, open('ET_Model.pkl','wb'))


# _**Loading the pickle file with the model name**_

# In[61]:


model=pickle.load(open('ET_Model.pkl','rb'))


# _**Predicting the dependent variable using the loaded pickle file and getting the r2 score and best models of the loaded pickle file.**_

# In[62]:


y_pred=model.predict(x_test)
display(r2_score(y_test,y_pred))
print(model)

