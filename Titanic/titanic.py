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
# print(df['Ticket'].value_counts(dropna=False))

df=df[['Survived','Ticket','Pclass','Age','SibSp','Parch','Fare','Embarked']]

df['Embarked']=df['Embarked'].fillna("S")
y=df[['Survived']].values
x=pd.factorize(df['Embarked'])
df['Embarked']=x[0]
x=pd.factorize(df['Ticket'])
df['Ticket']=x[0]

# print(df['Age'].value_counts(dropna=False))
df['family_size']=df['SibSp'] + df['Parch'] + 1

df=df.values

solver=MICE()
df=solver.complete(df)
# a=np.delete(df,[0,1,2,3,4,5,6],axis=1)
# plt.plot(a,y,'ro')
# plt.show()

# print(np.isnan(df).sum())

test=pd.read_csv('test.csv')
test.Fare[152]=test['Fare'].median()
test['family_size']=test['SibSp'] + test['Parch'] + 1
test_data=test[['Ticket','Pclass','Age','SibSp','Parch','Fare','Embarked','family_size']]

x=pd.factorize(test_data['Embarked'])
test_data['Embarked']=x[0]
x=pd.factorize(test_data['Ticket'])
test_data['Ticket']=x[0]

print(test_data['Age'].value_counts(dropna=False))
test_features=test_data.values
test_features=solver.complete(test_features)

x=np.delete(df,0,axis=1)

model=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=10,min_samples_split=10)
model=model.fit(x,y)


my_prediction = model.predict(test_features)

PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
# print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


# print(type(x[5][0]))



#https://campus.datacamp.com/courses/kaggle-python-tutorial-on-machine-learning/predicting-with-decision-trees?ex=2
