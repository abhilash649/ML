import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
y=train[['p']].values
del train['p']

x=train.values

model=tree.DecisionTreeClassifier(criterion='gini')
# model=RandomForestClassifier()
model=model.fit(x,y)



b=test[['p']].values
del test['p']

a=test.values

c=model.predict(a)

# print(x)
print "\nAccuracy of Model without Imputation = ",accuracy_score(b,c)



# cols=['p','x','s','n','t','f','c','k','e.1','w','o','u']