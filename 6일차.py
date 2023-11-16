import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


df1 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-02.csv')

df1.info()
df2.info()

df1['blue'] = df1['blue'].astype('category')
df1['dual_sim'] = df1['dual_sim'].astype('category')
df1['four_g'] = df1['four_g'].astype('category')
df1['n_cores'] = df1['n_cores'].astype('category')
df1['three_g'] = df1['three_g'].astype('category')
df1['touch_screen'] = df1['touch_screen'].astype('category')
df1['price_range'] = df1['price_range'].astype('category')

df2['blue'] = df2['blue'].astype('category')
df2['dual_sim'] = df2['dual_sim'].astype('category')
df2['four_g'] = df2['four_g'].astype('category')
df2['n_cores'] = df2['n_cores'].astype('category')
df2['three_g'] = df2['three_g'].astype('category')
df2['touch_screen'] = df2['touch_screen'].astype('category')
df2 = df2.drop('ID', axis = 1)

y = df1['price_range']
x = pd.get_dummies(df1)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(f1_score(y_valid, pred, average = 'macro'))

x_test = pd.get_dummies(df2)

pred = model.predict(x_test)

result = pd.DataFrame({'pred' : pred})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')
print(a)