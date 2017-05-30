import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split



df=pd.read_csv('mushroom.csv')

df['e.1'][df['e.1']=='?']=np.nan	
# df=df[['p','x','s','n','t','f','c','k','e.1','w','o','u']]
# cols=['p','x','s','n','t','f','c','k','e.1','w','o','u']
# df=df.dropna()

df=df[['p','x','s','n','c','e.1']]
cols=['p','x','s','n','c','e.1']

for x in cols:
	s=pd.factorize(df[x])
	df[x]=s[0]


train,test=train_test_split(df,test_size=0.2)


train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)





''' u'p', u'x', u's', u'n', u't', u'p.1', u'f', u'c', u'n.1', u'k', u'e',
       u'e.1', u's.1', u's.2', u'w', u'w.1', u'p.2', u'w.2', u'o', u'p.3',
       u'k.1', u's.3', u'u'  '''