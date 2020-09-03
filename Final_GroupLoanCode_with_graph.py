
# coding: utf-8

# In[24]:


import numpy as np


# In[39]:


import pandas as pd
#import keras
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import xgboost as xgb
#from tensorflow.keras.preprocessing.text import one_hot
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics

df = pd.read_csv('GroupConciseML.csv')
#df['max_age'] = df['max_age']/df['age']
#df['min_age'] = df['min_age']/df['age']
#df['max_familymemberscount'] = df['max_familymemberscount']/df['familymemberscount']
#df['min_familymemberscount']=df['min_familymemberscount']/df['familymemberscount']
df=df[df['groupSize']>2]

df["default"] = [1 if x > 0 else 0 for x in df["everDefault"]]
df["npa"] = [1 if x > 0 else 0 for x in df["everNPA"]]

trainOn = "default"
insteadOf = "npa"
 
columnsToIgnore = [ insteadOf, "var_age","everDefault","logAvgAssets","var_familymemberscount","ownAsset","avg_expenditure","maritalStatus", "everNPA","spouseeducationalqualification","casteSimilarityProp","var_totalLand","maxRelCatProp"]

columnsToIgnore = columnsToIgnore

trainData = df.drop(columnsToIgnore, axis=1)
# Define xVar and yVar
yVar = trainData[trainOn]
xVar = trainData.drop([trainOn, "groupName"], axis=1)

#X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.3, random_state=100)
xgb_model = xgb.XGBClassifier()





# Tuning to get best fir parameters
params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(50, 150), # default 100
    "subsample": uniform(0.6, 0.4)
         }


#{'colsample_bytree': 0.9956108181311257, 'gamma': 0.08230004105331823,
#'learning_rate': 0.08839363504912673, 'max_depth': 3, 'n_estimators': 148, 'subsample': 0.7086273122748417}

# Run the model


# Performance
'''print( "Precision : ", metrics.average_precision_score(y_test, preds) )
print( "Recall : ", metrics.recall_score(y_test, preds) )

print(confusion_matrix(y_test, preds))
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

y_score = xg_reg.predict_proba(X_test)#['1']
y_scores = y_score[:,1]
print( "ROC AUC Score : ", roc_auc_score(y_test, y_scores) )

print( "F1 Score : ", metrics.f1_score(y_test, preds) )

print("Gini : ", 2*roc_auc_score(y_test, y_scores)-1)'''



# Curtail num of 0s in training
trainAllPositive = trainData[trainData[trainOn]==1]
trainAllNegative = trainData[trainData[trainOn]==0]

trainAllNegativeSample = trainAllNegative.sample(frac = 0.32)
#XVar_new1 = trainAllPositive.drop([ 'groupName'], axis=1).append(trainAllNegativeSample.drop([ 'groupName'], axis=1))
XVar_new = trainAllPositive.drop(['default1', 'groupName'], axis=1).append(trainAllNegativeSample.drop(['default1', 'groupName'], axis=1))
yVar_new = trainAllPositive[trainOn].append(trainAllNegativeSample[trainOn])



X_train, X_test, y_train, y_test = train_test_split(XVar_new, yVar_new, test_size=0.3, random_state=123)
X_train.drop(['default'],inplace=True,axis=1)
X_test_without_default=X_test.drop(['default'],axis=1)


search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=23, n_iter=100, cv=3, verbose=0, n_jobs=1, return_train_score=True)
search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)


xg_reg = xgb.XGBClassifier(params=search.best_params_)

xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test_without_default)

print( "Precision : ", metrics.average_precision_score(y_test, preds) )
print( "Recall : ", metrics.recall_score(y_test, preds) )

print(confusion_matrix(y_test, preds))

y_score = xg_reg.predict_proba(X_test_without_default)#['1']
y_scores = y_score[:,1]
print( roc_auc_score(y_test, y_scores) )
print( "F1 Score : ", metrics.f1_score(y_test, preds) )
print("Gini : ", 2*roc_auc_score(y_test, y_scores)-1)


# In[40]:


y_score = xg_reg.predict_proba(X_test_without_default)
y_scores = y_score[:,1]
y_scores=pd.DataFrame(y_scores)
y_scores.to_csv('yProb.csv')


# In[41]:


X_test.index= range(len(X_test['default']))
X_test


# In[42]:


for_graph=pd.concat([X_test['default'],y_scores],axis=1)


# In[43]:


for_graph.rename(columns={0:'Estimated_Default_Rate','default':'Observed_Default_Rate'}, inplace=True)


# In[44]:


d=for_graph.groupby(pd.qcut(for_graph.Estimated_Default_Rate,25)).sum() #for total default in a group
c=for_graph.groupby(pd.qcut(for_graph.Estimated_Default_Rate,25)).mean() #for getting mean probability of default in that group


# In[45]:


z = pd.concat([c['Estimated_Default_Rate'],d['Observed_Default_Rate']],axis=1)


# In[46]:


z


# In[47]:


from matplotlib.pylab import plt


# In[48]:


y=np.arange(1,26)
z['Observed_Default_Rate']=z['Observed_Default_Rate']/(z['Observed_Default_Rate'].max())


# In[49]:


plt.plot(y,z['Estimated_Default_Rate'],color='r')
plt.plot(y,z['Observed_Default_Rate'],color='b')

plt.show()
'''blue line is for Observed and red line is for estimated default rate, here as we can see the trend is similar'''


# In[50]:


import seaborn as sns; sns.set(context='paper', style='darkgrid', palette='muted', font='sans-serif', font_scale=1.8, color_codes=True, rc=None)
#tips = pd.read_excel('Book1.xlsx')
#sns.color_palette("#95a5a6")
g = sns.lmplot(x="Estimated_Default_Rate", y="Observed_Default_Rate", data=z)


# Same Graph but with pearsonR coefficient 

# In[51]:


sns.jointplot(z['Estimated_Default_Rate'], z['Observed_Default_Rate'], kind="reg")


# In[52]:


z.to_csv('final.csv') 

