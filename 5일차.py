from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P210204-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P210204-02.csv')

df1['Reached.on.Time_Y.N'] = df1['Reached.on.Time_Y.N'].astype('category')
y = df1['Reached.on.Time_Y.N']
df1 = df1.drop('Reached.on.Time_Y.N', axis = 1)
print(df1)
x = df1.drop('ID', axis = 1)
print(x)
x = pd.get_dummies(x)
print(x)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3)

model = RandomForestClassifier(n_estimators = 300)
model.fit(x_train, y_train)
pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))

test_id = df2['ID']
x_test = df2.drop('ID', axis = 1)
print(x_test)
x_test = pd.get_dummies(x_test)
print(x_test)
pred_test = model.predict_proba(x_test)

print(pred_test)

result = pd.DataFrame({"ID" : test_id, "pred_proba" : pred_test[:,1]})
result.to_csv("수험번호.csv", index=False)
