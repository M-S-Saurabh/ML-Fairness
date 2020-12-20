#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn import svm
from sklearn.utils import resample

from scipy.stats import pearsonr


# In[2]:


get_ipython().run_line_magic('cd', 'Downloads')


# In[116]:


get_ipython().system('pip install sklego')

get_ipython().run_cell_magic('', '', '# sensitive attribute: gender(male/female) or gender(male/female) and race(white/non-white)')

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']

# the files for the same are in data/adult.zip, please extract the zio file and place them in data folder. In below lines put path for reading csv 
train = pd.read_csv('adult.data', sep=",\s", header=None, names = column_names, engine = 'python')
test = pd.read_csv('adult.test', sep=",\s", header=None, names = column_names, engine = 'python')
test['income'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')


adult = pd.concat([test,train])
adult.reset_index(inplace = True, drop = True)
adult.drop(0, inplace=True)
# adult


# In[5]:


print('Gender vs Income')

gender = round(pd.crosstab(adult.gender, adult.income).div(pd.crosstab(adult.gender, adult.income).apply(sum,1),0),2)
gender.sort_values(by = '>50K', inplace = True)
ax = gender.plot(kind ='bar', title = 'Proportion distribution across gender levels')
ax.set_xlabel('Gender level')
ax.set_ylabel('Proportion of population')


# In[6]:


adult[(adult['gender'] == "Female") & (adult['income'] == '<=50K')].count()


# In[7]:


# adult.age.unique()
print(adult.workclass.unique().tolist())




print(adult.gender.unique())


# In[9]:


print(adult.age.unique())


# In[10]:


print(adult.education.unique())


# In[11]:


print(adult['marital-status'].unique())


# In[12]:


print(adult.occupation.unique())


# In[13]:


print(adult['native-country'].unique())


# In[14]:


print(adult.relationship.unique())


# In[15]:


# Gender Bias (Train + Test)
print(adult.gender.unique())
print(adult['gender'].value_counts())
# adult['gender'].value_counts().plot.bar()
plt.show()


# In[16]:


print(adult['income'].value_counts().plot.bar())
plt.show()


# In[188]:


# adult.describe()


# indicating whether theincome of a person exceeds $50K/yr or not. Used in fairness-related studies 
# that want to compare gender or race inequalities based on people’s annual incomes, or various other studies

# In[385]:


data = np.load("adult/adult_train.npz")
train_x = data['x']
train_y = data['y']
train_a = data['a']
print('train_x.shape',train_x.shape)
print('train_y.shape',train_y.shape)
print('train_a.shape',train_a.shape)


# In[386]:


test_data = np.load("adult/adult_test.npz")
test_x = test_data['x']
test_y = test_data['y']
test_a = test_data['a']
print('test_x.shape',test_x.shape)
print('test_y.shape',test_y.shape)
print('test_a.shape',test_a.shape)


# In[387]:


headers = pd.read_csv("adult/adult_headers.txt", sep=" ",header=None)
headers = np.array(headers)


# In[388]:


train_x_pd = pd.DataFrame(data = train_x, columns = headers[0:113,0], dtype='int')
train_y_pd = pd.DataFrame(data = train_y, columns = np.array(headers[113]), dtype='int')

test_x_pd = pd.DataFrame(data = test_x, columns = headers[0:113,0], dtype='int')
test_y_pd = pd.DataFrame(data = test_y, columns = np.array(headers[113]), dtype='int')


# In[389]:


# check if correct column number 
# test_x_pd['sex_Male'][0]

test_x_pd['sex_Male']


# In[390]:


# test_x_pd['sex_Female']
# below checks the number is similar or not 
(test_x_pd['sex_Female'] != 0).sum(), (test_x_pd['sex_Male'] == 0).sum()


# In[391]:


# get column number of a column name
test_x_pd.columns.get_loc('sex_Male'), test_x_pd.columns.get_loc('sex_Female')


# In[392]:


# list(test_x_pd.iloc[0])


# In[393]:


# verify DF conversion to array is consistent
print(test_x[0][67])


# In[394]:


# this has gender column
print(train_x_pd.head())


# In[395]:


# 1 - False, 0 - True
df = pd.DataFrame(train_x_pd['sex_Female'].tolist())
(train_x_pd['sex_Female'] == 0).sum(axis=0), (train_x_pd['sex_Male'] == 1).sum(axis=0)
df1 = pd.DataFrame(train_x_pd['sex_Female'].tolist())
df1.columns = ['gender']
df2 = pd.DataFrame(train_y_pd['income'].tolist())
df2.columns = ['income']
df3 = pd.concat([df1, df2],axis=1)
print(df3)
print(df3[(df3['gender'] == 0) & (df3['income'] == 0)].count())


# In[396]:


print(train_x_pd.columns.tolist())


# In[397]:


# 1 - greater than 50k, 0 - less than 50k
# df = train_y_pd['income'].value_counts().reset_index()
# df.columns = ['income', 'count']
# df


# In[398]:


tra_pd = pd.DataFrame(data = train_a, columns = np.array(headers[113]),dtype='int')
# tra_pd.head()
# (tra_pd != 0).sum(1)
(tra_pd==0).sum(axis=0)


# In[399]:


from pandas.util.testing import assert_frame_equal
print(tra_pd.equals(test_y_pd))


# In[400]:


#income unbalance
more_50k = np.count_nonzero(train_y)
less_50k = len(train_y)-np.count_nonzero(train_y)
print('more than 50k',more_50k)
print('less than 50k',less_50k)
plt.bar(['less thank 50k','more thank 50k'],[less_50k,more_50k] , color=['black', 'red'])
plt.xlabel("Income")
plt.ylabel("Number of samples")
plt.show()


# In[401]:


def accuracy (y,y_hat):
    count=0
    for i in range (len(y_hat)):
        if y[i] == y_hat[i]:
            count+=1
    accuracy= count/len(y_hat)*100
    print('accuracy={0:.2f}%'.format(accuracy))
    
def reweighted_accuracy(y,y_hat,A):
    count_A0 =0
    count_A1 =0
    n_A1 = np.count_nonzero(A)
    n_A0 = len(A) - np.count_nonzero(A)
    for i in range (len(y_hat)):
        if y[i] == y_hat[i]:
            if A[i] == 1:
                count_A1 +=1
            else:
                count_A0 +=1
    accuracy = 0.5* (count_A1/n_A1 + count_A0/n_A0) *100
    print('re-weighted accuracy={0:.2f}%'.format(accuracy))


# In[402]:


def DP_accuracy (y_hat,A):
    A , y_hat = A.reshape(-1) , y_hat.reshape(-1)
    sum_A0 = 0
    sum_A1 = 0
    n_A1 = np.count_nonzero(A)
    n_A0 = len(A) - np.count_nonzero(A)
    for i in range (len(y_hat)):
        sum_A0 += y_hat[i]*(1-A[i])
        sum_A1 += y_hat[i]*A[i]
    accuracy= abs(sum_A0/n_A0 - sum_A1/n_A1)
    print('DP accuracy={0:.2f}'.format(accuracy))
    return np.abs((sum_A0/n_A0) - (sum_A1/n_A1))


# In[403]:


def all_accuarcy_fun (y, y_hat , A):
    y, y_hat, A = np.array(y), np.array(y_hat), np.array(A)
    accuracy (y,y_hat)
#     reweighted_accuracy (y,y_hat,A)
    DP_accuracy (y_hat,A)


# In[404]:


def classifier (train_x , train_y , test_x, test_y, test_a):
    
    print('-------------- Linear SVM --------------')
    linear_svm = svm.LinearSVC(dual=False).fit(train_x, train_y.ravel())
    a_hat = linear_svm.predict(test_x)
    all_accuarcy_fun(test_y , a_hat , test_a)
    
    print('-------------- Logistic Regression --------------')
    logistic_after_removing_A = LogisticRegression(max_iter=7000).fit(train_x, train_y.ravel())
    a_hat = logistic_after_removing_A.predict(test_x)
    all_accuarcy_fun(test_y , a_hat ,test_a)

    print('-------------- SVC --------------')
    svc = SVC(gamma='auto').fit(train_x, train_y.ravel())
    a_hat = svc.predict(test_x)
    all_accuarcy_fun(test_y , a_hat , test_a)


# Predict for Income

# In[405]:


print(classifier(train_x, train_y, test_x, test_y , test_a))


# In[406]:


# just for safety but not required
test_x_pd.drop(['income\n'], axis = 1, inplace = True, errors = 'ignore')


# In[407]:


from sklearn.linear_model import LogisticRegression
clf_LR=LogisticRegression(max_iter=7000)
clf_LR=clf_LR.fit(train_x, train_y)
# print("\n Training score: ",clf_LR.score(train_x, train_y)) #evaluating the training error
# predictions = clf_LR.predict(x_val)

y_hat=clf_LR.predict(test_x_pd)
j = clf_LR.predict_proba(test_x_pd)
print(j)
 #metrics.accuracy_score(pred,y_val)
# sc =  accuracy(test_y, y_hat)
# print("\nThe accuracy score: ",sc)


# In[408]:


dp = DP_accuracy(y_hat, test_a)
print("\nThe demographic parity : ",dp)


# In[409]:


# False Positive Rate
import numpy as np
from sklearn.metrics import confusion_matrix

# y_pred = prediction 
# y_true = ground truth 

def statistics_metrics(y_true, y_pred):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    print(cnf_matrix)


    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

#     print("FPR",FPR)
    return FPR, TPR


# In[410]:


fpr, tpr = statistics_metrics(test_y,y_hat)
print(fpr)


# In[411]:


# test_x[0][67]


# In[412]:


# 67 columns has sex_Male attribute
columns = 113
test_x_male_list, test_x_female_list = [], []
for num, x in enumerate(test_x):
    if x[67] == 0.0:
        test_x_male_list.append(num)
    else:
        test_x_female_list.append(num)
    
# verify number 
print(len(test_x_male_list), len(test_x_female_list))


# In[413]:


# extracting test data y by gender
testy_male = []
for num, y in enumerate(test_y):
    if num in test_x_male_list:
        if num == test_x_male_list[test_x_male_list.index(num)]:
            testy_male.append(y)
print(len(testy_male)) 

testy_female = []
for num, y in enumerate(test_y):
    if num in test_x_female_list:
        if num == test_x_female_list[test_x_female_list.index(num)]:
            testy_female.append(y)
print(len(testy_female)) 


# In[414]:


# extracting tpr, fpr by gender
# 1 - greater than 50k, 0 - less than 50k
y_male_pred = []
for num, y_p in enumerate(y_hat):
    if num in test_x_male_list:
        if num == test_x_male_list[test_x_male_list.index(num)]:
            y_male_pred.append(y_p)
print(len(y_male_pred))  

y_female_pred = []
for num, y_p in enumerate(y_hat):
    if num in test_x_female_list:
        if num == test_x_female_list[test_x_female_list.index(num)]:
            y_female_pred.append(y_p)
print(len(y_female_pred)) 


fpr_f, tpr_f = statistics_metrics(testy_female, y_female_pred)
fpr_m, tpr_m = statistics_metrics(testy_male, y_male_pred)

fpr_list, tpr_list = [], []

fpr_list.append(fpr_f[1])
fpr_list.append(fpr_m[1])

tpr_list.append(tpr_f[1])
tpr_list.append(tpr_m[1])
# print(fpr_f, fpr_m, tpr_f)


# In[415]:


def plot_fpr(columns, fpr_list):
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(columns)]
    plt.bar(x_pos, fpr_list, color='green')
    plt.xlabel("Factors")
    plt.ylabel("False Positive Rate")
    plt.title("FPR variation by gender")
    plt.xticks(x_pos, columns)
    plt.show()
    
def plot_tpr(columns, tpr_list):
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(columns)]
    plt.bar(x_pos, tpr_list, color='green')
    plt.xlabel("Factors")
    plt.ylabel("True Positive Rate")
    plt.title("TPR variation by gender")
    plt.xticks(x_pos, columns)
    plt.show()


# In[416]:


plot_tpr(["Female", "Male"],tpr_list)


# In[417]:


plot_fpr(["Female", "Male"],fpr_list)


# In[418]:


# extract test_a by gender 
testa_male = []
for num, y in enumerate(test_a):
    if num in test_x_male_list:
        if num == test_x_male_list[test_x_male_list.index(num)]:
            testa_male.append(y_p)
print(len(testa_male))


testa_female = []
for num, y in enumerate(test_a):
    if num in test_x_female_list:
        if num == test_x_female_list[test_x_female_list.index(num)]:
            testa_female.append(y_p)
print(len(testa_female))


# In[422]:


testy_female = np.array(testy_female)
test_y_male = np.array(testy_male)

y_female_pred = np.array(y_female_pred)
y_male_pred  = np.array(y_male_pred)

print(testy_female.shape, test_y_male.shape)


# In[429]:


dp_f = DP_accuracy(y_female_pred,testy_female)
dp_m = DP_accuracy(y_male_pred, test_y_male)
dp_list = [dp_f, dp_m]
# dp_f, dp_m


# In[431]:


def plot_dp(columns, dp_list):
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(columns)]
    plt.bar(x_pos, dp_list, color='red')
    plt.xlabel("Factors")
    plt.ylabel("Demographic Parity")
    plt.title("Demographic Parity variation by gender")
    plt.xticks(x_pos, columns)
    plt.show()

plot_dp(['Female', 'Male'],dp_list )


# In[432]:


# FPR for income 


fig = plt.figure()
# fig = plt.style.use('ggplot')
ax = fig.add_axes([0,0,1,1])
xlabel = ["less than 50k", "greater than 50k"]
ax.bar(xlabel,fpr,color=['blue'])
plt.show()
# plt.bar([0,1], fpr, )


# In[ ]:


# FPR for income 
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
xlabel = ["less than 50k", "greater than 50k"]
ax.bar(xlabel,tpr,color=['blue'])
plt.show()
# plt.bar([0,1], fpr, )

# References: 
# https://github.com/AissatouPaye/Fairness-in-Classification-and-Representation-Learning
# https://arxiv.org/pdf/2001.09784.pdf
# https://rdrr.io/cran/dslabs/man/admissions.html
# https://developers.google.com/machine-learning/crash-course/fairness/evaluating-for-bias
# https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb
# https://fairmlbook.org/code/adult.html
# https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/adult_dataset.py
# https://arxiv.org/pdf/1908.09635.pdf
# https://arxiv.org/pdf/1809.09245.pdf
# https://fairmlclass.github.io/#sources
# https://arxiv.org/pdf/1908.09635.pdf
# http://cs.carleton.edu/cs_comps/1920/fairness/index.php
# https://github.com/montaserFath
# https://closedloop.ai/a-new-metric-for-quantifying-fairness-in-healthcare/
# https://github.com/rehmanzafar/dlime_experiments
# https://fairmlbook.org/pdf/fairmlbook.pdf
# https://fairmlclass.github.io/
# http://www.cs.cornell.edu/~hubert/files/publications/silva_chi.pdf





