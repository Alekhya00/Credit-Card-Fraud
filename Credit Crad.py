#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
import numpy as np
# import keras
np.random.seed(2)


# In[116]:


data = pd.read_csv('creditcard.csv')


# In[117]:


data.head()


# In[118]:


from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis = 1)


# In[119]:


data = data.drop(['Time'],axis = 1)
data.head()


# In[120]:


# x = data.iloc[:, data.columns != 'class']
# y = data.iloc[:, data.columns == 'class']
columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]
target = "Class"
x = data[columns]
y = data[target]


# In[121]:


x.head()
print(x.shape)


# In[122]:


y.head()


# In[123]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)


# In[124]:


x_train.shape


# In[125]:


x_test.shape


# In[126]:


y_train.shape


# In[127]:


from sklearn.ensemble import RandomForestClassifier


# In[128]:


random_forest = RandomForestClassifier(n_estimators  = 100)


# In[129]:


random_forest.fit(x_train,y_train.values.ravel())


# In[130]:


y_pred = random_forest.predict(x_test)


# In[131]:


random_forest.score(x_test,y_test)


# In[132]:


import matplotlib.pyplot as plt
import itertools
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
# def plot_confusion_matrix(cm,classes,normalize = False,title = 'Confusion matrix',cmap = plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float')/cm.sum(axis = 1)[:,np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print("Confusion matrix, without normalization")
#     print(cm)
#     plt.imshow(cm,interpolation = 'nearest',cmap = cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks,classes, rotation = 45)
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max()/2.
#     for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
#         plt.text(j,i format(cm[i,j]),fmt),horizontalalignment  = "center",color = "white" if cm[i,j]> thresh else "black"
#     plt.ylabel('True label') 
#     plt.xlabel('Predicted label')
#     plt.tight_layout()


# In[135]:


cnf_matrix = confusion_matrix(y_test,y_pred)
print(cnf_matrix)


# In[102]:


# plot_confusion_matrix(cnf_matrix,classes = [0,1])


# In[137]:


ypred = random_forest.predict(x)


# In[138]:


cnf_matrix = confusion_matrix(y,ypred.round())


# In[140]:


Fraud = data[data['Class']==1]

Valid = data[data['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))


# In[141]:


print(outlier_fraction)

print("Fraud Cases : {}".format(len(Fraud)))

print("Valid Cases : {}".format(len(Valid)))


# In[142]:


from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix


# In[143]:


n_outliers = len(Fraud)
# n_errors = (y_pred != y_test).sum()
# acc = accuracy_score(y_test,y_pred)
# print(accuracy_score(y,y_pred))
# print(precision_score(y,y_pred))
# print(recall_score(y,y_pred))
# print(f1_score(y,y_pred))
n_errors = (ypred != y).sum()
    # Run Classification Metrics
# print("{}: {}".format(clf_name,n_errors))
print("Accuracy Score :")
print(accuracy_score(y,ypred))
print("Classification Report :")
print(classification_report(y,ypred))


# In[ ]:




