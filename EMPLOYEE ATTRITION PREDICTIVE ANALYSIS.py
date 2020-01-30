#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install scikit-learn==0.19.1


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


plt.rc("font", size=14)


# In[3]:


#import excel file - xlsx
employee_sheets = pd.read_excel('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx', None)
employee_sheets


# In[4]:


#different sheet names
employee_sheets.keys()


# In[5]:


#convert sheets from xlsx to csv
employee_sheets['Existing employees'].to_csv('Existing Employees.csv')
employee_sheets['Employees who have left'].to_csv('Employees Left.csv')


# In[6]:


existing_employee_orig = pd.read_csv('Existing Employees.csv')
left_employee_orig = pd.read_csv('Employees Left.csv')


# In[7]:


existing_employee_orig.head()


# In[8]:


left_employee_orig.head()


# In[9]:


#check for missing data in existing employee records
existing_employee_orig.isna().any()


# In[10]:


#check for missing data in left employee records
left_employee_orig.isna().any()


# In[11]:


#create copies of the original datasets
existing_employee = existing_employee_orig
left_employee = left_employee_orig

#create new columns in each dataset: 1(yes/left) , 0(no/exists)
existing_employee['Employee Absence'] = '0'
left_employee['Employee Absence'] = '1'


# In[12]:


existing_employee.head()


# In[13]:


left_employee.head()


# In[14]:


#check duplicates
existing_employee.duplicated().sum()


# In[15]:


left_employee.duplicated().sum()


# In[16]:


#left_employee[(left_employee['Employee Absence']) == 1 & (j>0.7)].sum()


# In[17]:


#concentrate on the sales department
left_sales = left_employee[left_employee['dept'] == 'sales']
existing_sales = existing_employee[existing_employee['dept'] == 'sales']


# In[18]:


left_sales.count()


# In[19]:


left_sales.head()


# In[20]:


existing_sales.head()


# In[ ]:





# In[21]:


#concatenate the sales datasets
employee_data = pd.concat([left_sales,existing_sales], ignore_index=True)
employee_data.head()


# In[22]:


employee_data.tail()


# In[23]:


employee_data.describe()


# In[24]:


print('satisfaction : : ', employee_data['satisfaction_level'].value_counts())
print('evaluation : : ', employee_data['last_evaluation'].value_counts())
print('projects : : ', employee_data['average_montly_hours'].value_counts())
print('time : : ', employee_data['time_spend_company'].value_counts())
print('accidents : : ', employee_data['Work_accident'].value_counts())
print('promotions : : ', employee_data['promotion_last_5years'].value_counts())


# In[25]:


#check for null/NaN values
employee_data.isna().any()


# In[26]:


employee_data.skew()


# In[27]:


employee_data.kurtosis()


# In[28]:


#remove work_accident column
employee_data = employee_data.drop(columns="Work_accident")


# In[29]:


#alter the salary values from text to numerical

print('salary levels : ',employee_data['salary'].unique())
employee_data['salary'].replace('low', 1, inplace =True)
employee_data['salary'].replace('medium', 2, inplace =True)
employee_data['salary'].replace('high', 3, inplace =True)


# In[30]:


employee_data.head()


# In[31]:


employee_data['salary'].value_counts()


# In[32]:


j = employee_data['Employee Absence']
sns.countplot(x=j, data=employee_data, palette='hls')
plt.show()


# In[33]:


sns.countplot(employee_data.salary, data=employee_data, palette='hls')
plt.show()


# In[34]:


#blue 0 = employed, yellow 1 = left
#picks = employee_data['satisfaction_level','last_evaluation','number_project','promotion_last_5years']
plot = sns.PairGrid(employee_data, vars = ['satisfaction_level','last_evaluation','number_project','time_spend_company','promotion_last_5years', 'salary'], hue='Employee Absence')
plot.map(plt.scatter);
plot.add_legend()


# from the above, the correlations are not as high to easily spot a trend. However, the following can be noticed:
#     - employees with a high evaluation scores left most, and all employees that had 6 or more projects all left, 
#        however they spent less than 10 years in the company and few had received a promotion in the last 5 years 

# In[35]:


jk = employee_data['Employee Absence']
pd.crosstab(employee_data.average_montly_hours,jk).plot(kind='bar',figsize=(19,12))
pd.crosstab(employee_data.satisfaction_level,jk).plot(kind='bar',figsize=(15,12))
pd.crosstab(employee_data.last_evaluation,jk).plot(kind='bar',figsize=(15,12))
pd.crosstab(employee_data.number_project,jk).plot(kind='bar',figsize=(15,12))
pd.crosstab(employee_data.time_spend_company,jk).plot(kind='bar',figsize=(15,12))
pd.crosstab(employee_data.promotion_last_5years,jk).plot(kind='bar',figsize=(15,12))
pd.crosstab(employee_data.salary,jk).plot(kind='bar',figsize=(15,12))


# In[36]:


employee_data.corr()


# In[ ]:





# In[37]:


employee_data.time_spend_company.hist()
plt.title('Histogram of Time Spent at X')
plt.xlabel('Time')
plt.ylabel('Frequency')


# the data is skewed toward the left however it is in moderation.

# In[38]:


employee_data.promotion_last_5years.hist()
plt.title('Histogram of Promotions')
plt.xlabel('Promotion')
plt.ylabel('Frequency')


# In[39]:


employee_data.satisfaction_level.hist()
plt.title('Histogram of satisfaction')
plt.xlabel('satisfaction')
plt.ylabel('Frequency')


# In[40]:


employee_data.number_project.hist()
plt.title('Histogram of Project')
plt.xlabel('Project')
plt.ylabel('Frequency')


# In[41]:


employee_data.salary.hist()
plt.title('Histogram of Salary')
plt.xlabel('Salary')
plt.ylabel('Frequency')


# In[42]:


sns.boxplot(x='Employee Absence', y='salary', data=employee_data)


# In[80]:


#train and test data : logistic regression

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score

X = np.array([[employee_data.satisfaction_level, employee_data.last_evaluation, employee_data.number_project,employee_data.average_montly_hours, employee_data.time_spend_company, employee_data.promotion_last_5years]])
y = np.array(employee_data['Employee Absence'])

X = X.reshape(1, -1)
y = y.reshape(1, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)


# In[67]:


#X = X.reshape(1, -1)
#y = y.reshape(1, -1)


# In[56]:


X.transpose()


# In[71]:


X.shape


# In[72]:


y.shape


# In[73]:


regression = LogisticRegression(C=100, random_state=20)
regression.fit(X_train, y_train)


# In[ ]:





# In[ ]:


#accuracy check
y_predict = regression.predict(X_test)


# In[ ]:





# In[ ]:


#decision tree

from scikitlearn import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy', random_state=100)
dtc.fit(X_train, y_train)

dtc_ypred = dtc.predict(X_test)


# In[ ]:


confusion_matrix(y_test, dtc_ypred)


# In[ ]:


accuracy_score(y_test, dtc_ypred)


# In[ ]:





# In[ ]:


#PREDICTIONS ON EXISTING SALES DATASET TO SEE WHO WILL MOST LIKELY LEAVE

#existing_sales for test


#drop work_accident column 
existing_sales.drop(columns='Work_Accident')
existing_sales.head()


# In[ ]:


eX = np.array([[existing_sales.satisfaction_level, existing_sales.last_evaluation, existing_sales.number_project,existing_sales.average_montly_hours, existing_sales.time_spend_company, existing_sales.promotion_last_5years]])
ey = np.array(existing_sales['Employee Absence'])


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

eX_train, eX_test, ey_train, ey_test = train_test_split(eX, ey, test_size=0.30, random_state=0)

sc = StandardScaler()
eX_train = sc.fit_transform(eX_train)
eX_test = sc.fit_transform(eX_test)

eRegression = LogisticREgression(random_state=0)
eRegression.fit(eX_train, ey_train)


# In[ ]:


ey_pred = eRegression.predict(eX_test)


# In[ ]:


accuracy_score(ey_test, ey_pred)

