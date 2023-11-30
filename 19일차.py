import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-02.csv')

df1['TravelInsurance'] = df1['TravelInsurance'].astype('category')
y = df1['TravelInsurance']
df1 = df1.drop('X', axis = 1)
df1 = df1.drop('TravelInsurance', axis = 1)
x = pd.get_dummies(df1)

test_id = df2['X']
df2 = df2.drop('X', axis = 1)
x_test = pd.get_dummies(df2)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))
print(f1_score(y_valid, pred))
print(accuracy_score(y_valid, pred))
print(precision_score(y_valid, pred))
print(recall_score(y_valid, pred))

pred_test = model.predict_proba(x_test)

result = pd.DataFrame({'ID' : test_id, 'predict_proba' : pred_test[:,1]})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

#########################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P220404-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P220404-02.csv')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df1['Segmentation'] = df1['Segmentation'].astype('category')
y = le.fit_transform(df1['Segmentation'])
print(le.classes_)
df1 = df1.drop('ID', axis = 1)
df1 = df1.drop('Segmentation', axis = 1)
x = pd.get_dummies(df1)

test_id = df2['ID']
df2 = df2.drop('ID', axis = 1)
x_test = pd.get_dummies(df2)

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3)

model = RandomForestClassifier(n_estimators=300)

model.fit(x_train, y_train)

pred = model.predict_proba(x_valid)

print(roc_auc_score(y_valid, pred, multi_class = 'ovr', average = 'macro'))

pred_test = model.predict(x_test)

print(pred_test)

pred_test = le.inverse_transform(pred_test)

print(pred_test)

result = pd.DataFrame({'ID' : test_id, 'pred' : pred_test})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

############################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P220504-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P220504-02.csv')

y = df1['price']
df1 = df1.drop('price', axis = 1)
x = pd.get_dummies(df1)

x_test = pd.get_dummies(df2)

feature = list(set(x.columns).intersection(set(x_test.columns)))
x = x[feature]
x_test = x_test[feature]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3)

model = RandomForestRegressor(n_estimators=300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(mean_squared_error(y_valid, pred, squared = False))

pred_test = model.predict(x_test)

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

##############################################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-02.csv')

df1['price_range'] = df1['price_range'].astype('category')
y = df1['price_range']
df1 = df1.drop('price_range', axis = 1)
x = pd.get_dummies(df1)

df2 = df2.drop('id', axis = 1)
x_test = pd.get_dummies(df2)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

model = RandomForestClassifier(n_estimators=300)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3)

model.fit(x_train, y_train)

pred = model.predict_proba(x_valid)

print(roc_auc_score(y_valid, pred, multi_class = 'ovo', average = 'macro'))
print(accuracy_score(y_valid, pred))

######################################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-02.csv')

df1['price_range'] = df1['price_range'].astype('category')
df1['blue'] = df1['blue'].astype('category')
df1['dual_sim'] = df1['dual_sim'].astype('category')
df1['four_g'] = df1['four_g'].astype('category')
df1['n_cores'] = df1['n_cores'].astype('category')
df1['three_g'] = df1['three_g'].astype('category')
df1['touch_screen'] = df1['touch_screen'].astype('category')
df1['wifi'] = df1['wifi'].astype('category')

y = df1['price_range']
df1 = df1.drop('price_range', axis = 1)
x = pd.get_dummies(df1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

model = RandomForestClassifier(n_estimators=300)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3)

model.fit(x_train, y_train)

pred = model.predict_proba(x_valid)

print(roc_auc_score(y_valid, pred, multi_class = 'ovo', average = 'macro'))

#######################################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P230605.csv', encoding = 'euc-kr')

answer_1 = round((len(df['코드'] == 4) / len(df)),3)
from scipy.stats import chisquare

a = pd.DataFrame(df.groupby('코드').size())

b = pd.DataFrame({'1' : [0.05], '2' : [0.1], '3':[0.05], '4' : [0.8]})

a = np.array(a.iloc[:,0])
b = np.array(b.iloc[0])

result = chisquare(a, b*len(df))

answer_2 = round(result.statistic, 3)
answer_3 = round(result.pvalue, 3)

print(answer_1, answer_2, answer_3)

#######################################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P230606.csv')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x = pd.get_dummies(df[['O3', 'Solar', 'Wind']])
y = df['Temperature']

lm = LinearRegression()

lm.fit(x, y)

print(lm.coef_)

answer_1 = round(lm.coef_[0], 3)

from scipy.stats import ttest_ind, shapiro

result = ttest_ind(df['Wind'], df['Temperature'])

answer_2 = round(result.pvalue, 3)

x_test = pd.DataFrame({'O3' : [10], 'Solar' : [90], 'Wind' : [20]})

pred = lm.predict(x_test)

answer_3 = round(pred[0], 3)

print(answer_1, answer_2, answer_3)