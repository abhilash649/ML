import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from fancyimpute import MICE
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')
df['Cabin']=df['Cabin'].str[0]
df['Name']=df['Name'].str.partition(',')[2]
df['Name']=df['Name'].str.partition('.')[0]

df=df[['Survived','Name','Ticket','Pclass','Age','SibSp','Parch','Fare','Embarked','Cabin']]

df['Embarked']=df['Embarked'].fillna("S")
y=df[['Survived']].values
x=pd.factorize(df['Embarked'])
df['Embarked']=x[0]
x=pd.factorize(df['Ticket'])
df['Ticket']=x[0]
x=pd.factorize(df['Cabin'])
df['Cabin']=x[0]
x=pd.factorize(df['Name'])
df['Name']=x[0]

df['family_size']=df['SibSp'] + df['Parch'] + 1

df=df.values

solver=MICE()
df=solver.complete(df)

test=pd.read_csv('test.csv')
test['Name']=test['Name'].str.partition(',')[2]
test['Name']=test['Name'].str.partition('.')[0]
test.Fare[152]=test['Fare'].median()
test['Cabin']=test['Cabin'].str[0]
test['family_size']=test['SibSp'] + test['Parch'] + 1
test_data=test[['Name','Ticket','Pclass','Age','SibSp','Parch','Fare','Embarked','Cabin','family_size']]

x=pd.factorize(test_data['Embarked'])
test_data['Embarked']=x[0]
x=pd.factorize(test_data['Ticket'])
test_data['Ticket']=x[0]
x=pd.factorize(test_data['Cabin'])
test_data['Cabin']=x[0]
x=pd.factorize(test_data['Name'])
test_data['Name']=x[0]

test_features=test_data.values
test_features=solver.complete(test_features)

x=np.delete(df,0,axis=1)

model=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=10,min_samples_split=10)
# cv = cross_validation.KFold(len(x), n_folds=10)
# results = []
# for traincv, testcv in cv:
# 	probas = model.fit(x[traincv], y[traincv]).predict_proba(x[testcv])
# 	results.append( model.score(x[testcv],y[testcv]) )	
# print(np.array(results).mean())
model=model.fit(x,y)


my_prediction = model.predict(test_features)

PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])

#https://campus.datacamp.com/courses/kaggle-python-tutorial-on-machine-learning/predicting-with-decision-trees?ex=2