from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import sklearn
dir(sklearn.metrics)

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P220504-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P220504-02.csv')
print(df1)
print(df2)
df1.info()
df2.info()

y = df1['price']
df1 = df1.drop('price', axis = 1)
x = pd.get_dummies(df1)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3)

model = RandomForestRegressor(n_estimators = 300)
model.fit(x_train, y_train)
pred = model.predict(x_valid)

print(mean_squared_error(y_valid, pred, squared=False))

x_test = pd.get_dummies(df2)
print(x)
print(x_test)

feature = list(set(x.columns).intersection(x_test.columns))
x_train_feature = x[feature]
x_test_feature = x[feature]

model = RandomForestRegressor(n_estimators = 300)
model.fit(x_train_feature, y)

pred = model.predict(x_test_feature)

print(pred)

result = pd.DataFrame({'pred' : pred})

result.to_csv('수험번호.csv', index = False)
a = pd.read_csv('수험번호.csv')
print(a)