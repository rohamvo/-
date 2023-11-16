from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P220404-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P220404-02.csv')

df1['Segmentation'] = df1['Segmentation'].astype('category')
# le = LabelEncoder()
# y = le.fit_transform(df1['Segmentation'])
y = df1['Segmentation']
print(y)
df1 = df1.drop('ID', axis = 1)
df1 = df1.drop('Segmentation', axis = 1)
x = pd.get_dummies(df1)

model = RandomForestClassifier(n_estimators = 300)
x_train, x_valid, y_train, y_valid = train_test_split(x,y, test_size = 0.3)

model.fit(x_train, y_train)

pred = model.predict(x_valid)
print(confusion_matrix(y_valid, pred, labels = ['A', 'B', 'C', 'D']))
print(pred)
print(y_valid)

rf_score = accuracy_score(y_valid, pred)

model = DecisionTreeClassifier(random_state = 2000)
model.fit(x_train, y_train)
pred = model.predict(x_valid)
dt_score = accuracy_score(y_valid, pred)

print(rf_score, dt_score)

model = RandomForestClassifier(n_estimators = 300)
model.fit(x_train, y_train)
pred = model.predict(x_valid)
print(accuracy_score(y_valid, pred))

df2 = df2.drop('ID', axis = 1)

x_test = pd.get_dummies(df2)

pred = model.predict(x_test)

idx = []
for i in range(1, len(x_test)+1) :
    idx.append(i)

result = pd.DataFrame({'ID' : idx , 'pred' : pred})
print(result)