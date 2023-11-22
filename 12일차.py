import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P210204-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P210204-02.csv')

df1.info()
df2.info()

df1 = df1.drop('ID', axis = 1)
df1['Reached.on.Time_Y.N'] = df1['Reached.on.Time_Y.N'].astype('category')
y = df1['Reached.on.Time_Y.N']
df1 = df1.drop('Reached.on.Time_Y.N', axis = 1)
x = pd.get_dummies(df1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))
print(accuracy_score(y_valid, pred))

test_id = df2['ID']
df2 = df2.drop('ID', axis = 1)
x_test = pd.get_dummies(df2)

pred_test = model.predict_proba(x_test)

result = pd.DataFrame({'ID' : test_id, 'pred' : pred_test[:,1]})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

##################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-02.csv')

df1 = df1.drop('X', axis = 1)
df1['TravelInsurance'] = df1['TravelInsurance'].astype('category')
y = df1['TravelInsurance']
df1 = df1.drop('TravelInsurance', axis = 1)
x = pd.get_dummies(df1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))
print(accuracy_score(y_valid, pred))

df2 = df2.drop('X', axis = 1)

x_test = pd.get_dummies(df2)

pred_test = model.predict(x_test)

idx = [i for i in range(1, len(pred_test) + 1)]

result = pd.DataFrame({'index' : idx, 'pred' : pred_test})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P220404-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P220404-02.csv')

df1.info()
df2.info()

df1 = df1.drop('ID', axis = 1)
df1['Segmentation'] = df1['Segmentation'].astype('category')
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

print(f1_score(y_valid, pred, average='macro'))

test_id = df2['ID']
df2 = df2.drop('ID', axis = 1)
x_test = pd.get_dummies(df2)

pred_test = model.predict(x_test)

result = pd.DataFrame({'ID' : test_id, 'pred' : pred_test})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

############################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P220504-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P220504-02.csv')

y = df1['price']
df1 = df1.drop('price', axis = 1)
x = pd.get_dummies(df1)

x_test = pd.get_dummies(df2)

feature = list(set(x.columns).intersection(x_test.columns))

x = x[feature]
x_test = x_test[feature]

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestRegressor(n_estimators = 300)

model.fit(x_train, y_train)

pred_valid = model.predict(x_valid)

print(mean_squared_error(y_valid, pred_valid, squared = False))

pred_test = model.predict(x_test)

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

####################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-02.csv')

df1['prcie_range'] = df1['price_range'].astype('category')
y = df1['price_range']
df1 = df1.drop('price_range', axis = 1)
df1['blue'] = df1['blue'].astype('category')
df1['dual_sim'] = df1['dual_sim'].astype('category')
df1['four_g'] = df1['four_g'].astype('category')
df1['n_cores'] = df1['n_cores'].astype('category')
df1['three_g'] = df1['three_g'].astype('category')
df1['touch_screen'] = df1['touch_screen'].astype('category')
df1['wifi'] = df1['wifi'].astype('category')
x = pd.get_dummies(df1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25, random_state = 50)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred_valid = model.predict(x_valid)

cm = confusion_matrix(y_valid, pred_valid)
print(precision_score(y_valid, pred_valid, average = 'macro'))
print(recall_score(y_valid, pred_valid, average = 'macro'))

print(f1_score(y_valid, pred_valid, average = 'macro'))
print(roc_auc_score(y_valid, pred_valid, multi_class = 'ovo'))
print(accuracy_score(y_valid, pred_valid))

######################################################