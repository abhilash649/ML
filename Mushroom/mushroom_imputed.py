import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fancyimpute import MICE

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train['e.1'][train['e.1']==-1]=np.nan
test['e.1'][test['e.1']==-1]=np.nan

solver=MICE( min_value=0.0,
    max_value=3.0)

y=train[['p']].values
train=train.values
train=solver.complete(train)

x=np.delete(train,0,axis=1)

model=tree.DecisionTreeClassifier(criterion='gini')
model=model.fit(x,y)

b=test[['p']].values
test=test.values
test=solver.complete(test)
a=np.delete(test,0,axis=1)

c=model.predict(a)

print "\nAccuracy of Model with Imputation done with MICE = ",accuracy_score(b,c)

# cols=['p','x','s','n','t','f','c','k','e.1','w','o','u']
