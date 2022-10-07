#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm


# # Task 3: Load Dataset

# In[3]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)
boston_df.head()


# In[4]:


boston_df.info()


# # Task 4: Descriptive Statistics

# In[5]:


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.boxplot(boston_df['MEDV'])


# In[ ]:


#Provide a bar plot for the Charles river variable. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)


# In[8]:


division_eval = boston_df.groupby('CHAS')[['MEDV']].mean().reset_index()
division_eval


# In[10]:


sns.set(style="whitegrid")
ax = sns.barplot(x="CHAS", y="MEDV", data=division_eval)


# In[11]:


#Provide a boxplot for the MEDV variable vs the AGE variable. (Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)


# In[12]:


boston_df.loc[(boston_df['AGE'] <= 35), 'age_group'] = '35 years and younger'
boston_df.loc[(boston_df['AGE'] > 35)&(boston_df['AGE'] < 70), 'age_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df['AGE'] >= 70), 'age_group'] = '70 years and older'


# In[13]:


boston_df.head()


# In[15]:


sns.set(style="whitegrid")
ax=sns.boxplot("age_group", y="MEDV", data=boston_df)


# In[27]:


#Provide a scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town. What can you say about the relationship?
ax = sns.scatterplot(x='INDUS',y='NOX',data=boston_df).set_title('Nitric oxide concentration vs proportion of non-retail business acres per town')


# In[43]:


# Relationship:tendence of increasing the number of non-retail business acres while NOX increases till a aprox limit of 18 (INDUS), seems to be a kind of saturation


# In[45]:


#Create a histogram for the pupil to teacher ratio variable
plt.hist(boston_df['PTRATIO'])


# # Task 5: use appropriate tests to answer the questions provided

# In[48]:


#Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)
#Null hypothesis: there is no difference in median value of houses bounded by the Charles river or not
#Alternative hypothesis: there is  difference in median value of houses bounded by the Charles river or not
scipy.stats.ttest_ind(boston_df[boston_df["CHAS"] == 1]["MEDV"],
                   boston_df[boston_df["CHAS"] == 0]["MEDV"], equal_var = True)


# In[17]:


#pvalue <0.05 => We reject the null hypothesis => there is significant difference in median value


# In[16]:


#Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)
#Null hypothesis: there is no difference in median value of houses MEDV for each proportion of owner occupied units build prior to 1940 (AGE)
#Alternative hypothesis: there is significant difference in median value of houses MEDV for each proportion of owner occupied units build prior to 1940 (AGE)
ax = sns.scatterplot(x="MEDV", y="AGE", data=boston_df)


# In[23]:


#First, separate the three samples (one for each age category) into a variable each.
thirtyfive_lower = boston_df[boston_df['age_group'] == '35 years and younger']['MEDV']
thirtyfive_seventy = boston_df[boston_df['age_group'] == 'between 35 and 70 years']['MEDV']
seventy_older = boston_df[boston_df['age_group'] == '70 years and older']['MEDV']


# In[24]:


#Now, run a ANOVA.
f_statistic, p_value = scipy.stats.f_oneway(thirtyfive_lower, thirtyfive_seventy, seventy_older)
print("F_Statistic: {0}, P-Value: {1}".format(f_statistic,p_value))


# In[25]:


#P-Value <0.05 => We reject the null hypothesis => there is significant difference in median value of houses MEDV for each proportion of owner occupied units build prior to 1940 (AGE)


# In[29]:


# Can we conclude that there is no relationship between Nitric oxide concentrations NOX and INDUS proportion of non-retail business acres per town? (Pearson Correlation)
# Since they are both continuous variables we can use a pearson correlation test and draw a scatter plot
ax = sns.scatterplot(x='INDUS',y='NOX',data=boston_df).set_title('Nitric oxide concentration vs proportion of non-retail business acres per town')
#Null hypothesis: there is no difference in INDUS depending on NOX value
#Alternative hypothesis: there is significant difference in INDUS depending on NOX value


# In[31]:


scipy.stats.pearsonr(boston_df['NOX'], boston_df['MEDV'])


# In[32]:


#P-Value <0.05 => We reject the null hypothesis => there is significant difference in INDUS depending on NOX value


# In[35]:


# What is the impact of an additional [weighted distance to the five Boston employment centres-DIS) on the MEDV-median value of owner occupied homes? (Regression analysis)
#Null hypothesis: there is no difference in MEDV depending on DIS
#Alternative hypothesis: there is no difference in MEDV depending on DIS
ax = sns.scatterplot(x='DIS',y='MEDV',data=boston_df).set_title('MEDV vs DIS')


# In[36]:


## X is the input variables (or independent variables)
X = boston_df['DIS']
## y is the target/dependent variable
y = boston_df['MEDV']
## add an intercept (beta_0) to our model
X = sm.add_constant(X) 

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()


# In[37]:


#p=0>0.05 => we can not reject the null hypothesis, we conclude there is no difference in MEDV depending on DIS


# In[ ]:




