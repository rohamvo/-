import pandas as pd
import numpy as np

df1 = pd.read_csv("C:/Users/82108/Desktop/data/P220404-01.csv")
df2 = pd.read_csv("C:/Users/82108/Desktop/data/P220404-02.csv")

df1.info()
df2.info()

df1['Segmentation'] = df1['Segmentation'].astype('category')
df1 = df1.drop('ID', axis = 1)
y = df1['Segmentation']
df1 = df1.drop('Segmentation', axis = 1)
x = pd.get_dummies(df1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(f1_score(y_valid, pred, average = 'macro'))

id = df2['ID']
df2 = df2.drop('ID', axis = 1)
x_test = pd.get_dummies(df2)

test_pred = model.predict(x_test)

result = pd.DataFrame({"ID" : id, "pred" : test_pred})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

##############################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv("C:/Users/82108/Desktop/data/P220504-01.csv")
df2 = pd.read_csv("C:/Users/82108/Desktop/data/P220504-02.csv")

df1.info()
df2.info()

y = df1['price']
df1 = df1.drop('price', axis = 1)
x = pd.get_dummies(df1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestRegressor(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(mean_squared_error(y_valid, pred, squared = False))

x_test = pd.get_dummies(df2)

inter_feature = list(set(x.columns).intersection(x_test.columns))

x_train_inter = x[inter_feature]
x_test_inter = x[inter_feature]

model = RandomForestRegressor(n_estimators = 300)
model.fit(x_train_inter, y)

pred_test = model.predict(x_test_inter)

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

#########################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv("C:/Users/82108/Desktop/data/P230604-01.csv")
df2 = pd.read_csv("C:/Users/82108/Desktop/data/P230604-02.csv")

df1.info()
df2.info()

df1['blue'] = df1['blue'].astype('category')
df1['dual_sim'] = df1['dual_sim'].astype('category')
df1['four_g'] = df1['four_g'].astype('category')
df1['three_g'] = df1['three_g'].astype('category')
df1['touch_screen'] = df1['touch_screen'].astype('category')
df1['wifi'] = df1['wifi'].astype('category')
df1['price_range'] = df1['price_range'].astype('category')

y = df1['price_range']
df1 = df1.drop('price_range', axis = 1)
x = pd.get_dummies(df1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(f1_score(y_valid, pred, average = 'macro'))

df2['blue'] = df2['blue'].astype('category')
df2['dual_sim'] = df2['dual_sim'].astype('category')
df2['four_g'] = df2['four_g'].astype('category')
df2['three_g'] = df2['three_g'].astype('category')
df2['touch_screen'] = df2['touch_screen'].astype('category')
df2['wifi'] = df2['wifi'].astype('category')

df2 = df2.drop('id', axis = 1)

x_test = pd.get_dummies(df2)

pred_test = model.predict(x_test)

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)