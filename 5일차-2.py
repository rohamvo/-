from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-02.csv')

df1.info()
df2.info()

y = df1['TravelInsurance'].astype('category')
df1 = df1.drop('TravelInsurance', axis = 1)
df1 = df1.drop('X', axis = 1)

x = pd.get_dummies(df1)

model = RandomForestClassifier(n_estimators = 300)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))

num = df2['X']
df2 = df2.drop('X', axis = 1)
x_test = pd.get_dummies(df2)

pred_proba = model.predict_proba(x_test)

idx = []
for n in range(1, len(pred_proba)+1) :
    idx.append(n)
result = pd.DataFrame({"index" : idx, "y_pred" : pred_proba[:,1]})

result.to_csv("수험번호.csv", index = False)

a = pd.read_csv('수험번호.csv')

print(a)