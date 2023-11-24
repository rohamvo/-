import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/bmw.csv')

df.info()

df = df.sort_values(by = 'price', ascending = False)

answer = df['year'].iloc[0]
print(answer)

########################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/WA_Fn-UseC_-Telco-Customer-Churn1.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/WA_Fn-UseC_-Telco-Customer-Churn2.csv')

df1.info()
df2.info()

# common_feature = set(df1.columns).intersection(set(df2.columns))

df1 = df1.drop('customerID', axis = 1)
df1['Churn'] = df1['Churn'].map({'Yes' : 1, 'No' : 0}).astype('category')
y = df1['Churn']
df1 = df1.drop('Churn', axis = 1)
x = pd.get_dummies(df1)
df2 = df2.drop('customerID', axis = 1)
x_test = pd.get_dummies(df2)

common_feature = list(set(x.columns).intersection(x_test.columns))

x = x[common_feature]
x_test = x_test[common_feature]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(accuracy_score(y_valid, pred))
print(roc_auc_score(y_valid, pred))
print(precision_score(y_valid, pred))
print(recall_score(y_valid, pred))
print(f1_score(y_valid, pred))

pred_test = model.predict(x_test)

pred_test = ['No' if p == 0 else 'Yes' for p in pred_test]

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

