import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P210204-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P210204-02.csv')

df1.info()
df2.info()

df1['Reached.on.Time_Y.N'] = df1['Reached.on.Time_Y.N'].astype('category')
df1 = df1.drop('ID', axis = 1)

y = df1['Reached.on.Time_Y.N']
df1 = df1.drop('Reached.on.Time_Y.N', axis = 1)
x = pd.get_dummies(df1)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))

ID = df2['ID']
df2 = df2.drop('ID', axis = 1)
x_test = pd.get_dummies(df2)

pred_p = model.predict_proba(x_test)

result = pd.DataFrame({"ID" : ID , "pred_p" : pred_p[:,1]})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

################################################################
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


df1 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-02.csv')

df1.info()
df2.info()

df1['TravelInsurance'] = df1['TravelInsurance'].astype('category')
y = df1['TravelInsurance']
df1 = df1.drop('TravelInsurance', axis = 1)
df1 = df1.drop('X', axis = 1)
x = pd.get_dummies(df1)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))

df2 = df2.drop('X', axis = 1)

x_test = pd.get_dummies(df2)

pred_p = model.predict_proba(x_test)

idx = []
for i in range(1, len(pred_p)+1) :
    idx.append(i)

result = pd.DataFrame({'index' : idx, 'y_pred' : pred_p[:,1]})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)
