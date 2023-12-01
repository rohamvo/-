import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P210204-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P210204-02.csv')

df1['Reached.on.Time_Y.N'] = df1['Reached.on.Time_Y.N'].astype('category')
y = df1['Reached.on.Time_Y.N']
df1 = df1.drop('Reached.on.Time_Y.N', axis = 1)
df1 = df1.drop('ID', axis = 1)
x = pd.get_dummies(df1)

df2 = df2.drop('ID', axis = 1)
x_test = pd.get_dummies(df2)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3, stratify = y)

model = RandomForestClassifier(n_estimators=500)

model.fit(x_train, y_train)

pred = model.predict_proba(x_valid)

pred_test = model.predict_proba(x_test)

result = pd.DataFrame({'pred' : pred_test[:,1]})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

################################################################################


import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P210304-02.csv')

df1['TravelInsurance'] = df1['TravelInsurance'].astype('category')
y = df1['TravelInsurance']
df1 = df1.drop('TravelInsurance', axis = 1)
df1 = df1.drop('X', axis = 1)
x = pd.get_dummies(df1)

df2 = df2.drop('X', axis =1)
x_test = pd.get_dummies(df2)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3, stratify = y)

model = RandomForestClassifier(n_estimators=300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))
print(f1_score(y_valid, pred))

pred_test = model.predict_proba(x_test)

idx = []
for i in range(1, len(pred_test)+1) :
    idx.append(i)

print(len(idx), len(pred_test))

result = pd.DataFrame({'index' : idx, "y_pred" : pred_test[:,1]})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

##################################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P220404-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P220404-02.csv')

df1['Segmentation'] = df1['Segmentation'].astype('category')
from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
y = df1['Segmentation']
# y = le.fit_transform(y)
df1 = df1.drop('Segmentation', axis = 1)
df1 = df1.drop('ID', axis = 1)
x = pd.get_dummies(df1)

test_id = df2['ID']
df2 = df2.drop('ID', axis = 1)
x_test = pd.get_dummies(df2)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3, stratify = y)

model = RandomForestClassifier(n_estimators=300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(accuracy_score(y_valid, pred))

################################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P220504-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P220504-02.csv')

y = df1['price']
df1 = df1.drop('price', axis = 1)
x = pd.get_dummies(df1)

x_test = pd.get_dummies(df2)

common_feature = list(set(x.columns).intersection(x_test.columns))

x = x[common_feature]
x_test = x_test[common_feature]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor(n_estimators=300)

x_train, x_valid, y_train, y_valid = train_test_split(x,y, test_size = 0.3)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(mean_squared_error(y_valid, pred, squared = False))

pred_test = model.predict(x_test)

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

#####################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-01.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/P230604-02.csv')

df1['price_range'] = df1['price_range'].astype('category')

y = df1['price_range']
from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = pd.get_dummies(y)
df1 = df1.drop('price_range', axis = 1)
x = pd.get_dummies(df1)

df2 = df2.drop('id', axis = 1)
x_test = pd.get_dummies(df2)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3, stratify = y)

model = RandomForestClassifier(n_estimators=300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(accuracy_score(y_valid, pred))

pred_test = model.predict(x_test)

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

##############################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P230605.csv', encoding = 'euc-kr')

four = len(df[df['코드'] == 4]) / len(df)
answer_1 = round(four, 3)

from scipy.stats import chisquare

rate = np.array([0.05, 0.1, 0.05, 0.8])
num = df.groupby('코드').size()

print(np.array(num.iloc[0:]))

result = chisquare(np.array(num.iloc[0:]), rate*len(df))

answer_2 = round(result[0], 3)
answer_3 = round(result[1], 3)

print(answer_1, answer_2, answer_3)

#################################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P230606.csv')

y = df['Temperature']
x = df[['O3', 'Solar', 'Wind']]
x = pd.get_dummies(x)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x, y)

answer_1 = round(model.coef_[0],3)

from scipy.stats import ttest_ind, levene

print(levene(df['Wind'], df['Temperature']))

t_result = ttest_ind(df['Wind'], df['Temperature'], equal_var = False)

answer_2 = round(t_result[1],3)

test = pd.DataFrame({'O3' : [10], 'Solar' : [90], 'Wind' : [20]})

pred_test = model.predict(test)

answer_3 = round(pred_test[0], 3)

print(answer_1, answer_2, answer_3)

###############################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P230606.csv')

y = df['Temperature']
x = df[['O3', 'Solar', 'Wind']]

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x, y)

answer_1 = round(model.coef_[0],3)

from scipy.stats import ttest_ind, levene

print(levene(df['Wind'], df['Temperature']))

t_result = ttest_ind(df['Wind'], df['Temperature'], equal_var = False)

answer_2 = round(t_result[1],3)

test = pd.DataFrame({'O3' : [10], 'Solar' : [90], 'Wind' : [20]})

pred_test = model.predict(test)

answer_3 = round(pred_test[0], 3)

print(answer_1, answer_2, answer_3)

##########################################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-1.csv')

train = df.iloc[0:int(len(df) * 0.7), :]

train = train.sort_values(by = 'price', ascending = False)

result = train[0:5]

answer = int(result['depth'].median())

print(answer)

# 62

#######################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-2.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

df = df.dropna(subset = 'TotalCharges')

mean = df['TotalCharges'].mean()
std = df['TotalCharges'].std()

result = df[(df['TotalCharges'] > (mean-(1.5*std))) & (df['TotalCharges'] < (mean+(1.5*std)))]

answer = int(result['TotalCharges'].mean())

print(answer)

# 1663

##################################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-3.csv')

am_1 = df[df['am'] == 1]
am_0 = df[df['am'] == 0]

am_1 = am_1.sort_values(by = 'hp')
am_0 = am_0.sort_values(by = 'hp')

mean_1 = am_1['mpg'].iloc[0:5].mean()
mean_0 = am_0['mpg'].iloc[0:5].mean()

answer = round(mean_1 - mean_0, 1)

print(answer)

# 8.4

##############################################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-1.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-2.csv')

df1['Churn'] = df1['Churn'].map({"Yes" : 1, 'No' : 0})
df1['Churn'] = df1['Churn'].astype('category')
y = df1['Churn']
df1 = df1.drop('Churn', axis = 1)
x = pd.get_dummies(df1)

df2['Churn'] = df2['Churn'].map({"Yes" : 1, 'No' : 0})
df2['Churn'] = df2['Churn'].astype('category')
y_test = df2['Churn']
df2 = df2.drop('Churn', axis = 1)
x_test = pd.get_dummies(df2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sklearn

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3, stratify = y)

model = RandomForestClassifier(n_estimators=300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))
print(accuracy_score(y_valid, pred))
print(f1_score(y_valid, pred))

pred_test = model.predict(x_test)

print(roc_auc_score(y_test, pred_test))
print(accuracy_score(y_test, pred_test))
print(f1_score(y_test, pred_test))

pred_test = ['No' if p == 0 else 'Yes' for p in pred_test]

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

#######################################################################################################

import pandas as pd
import numpy as np

before = np.array([200,210,190,180,175])
after = np.array([180,175,160,150,160])

from scipy.stats import shapiro, ttest_rel

result = ttest_rel(after, before, alternative = 'less')

answer_1 = round(result[0], 5)
answer_2 = round(result[1], 5)
answer_3 = ['기각' if answer_2 < 0.05 else '채택']

print(answer_1, answer_2, answer_3)

#######################################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-6.csv')

from scipy.stats import shapiro, ttest_1samp, wilcoxon

result = wilcoxon(df['Temp']-75, alternative = 'two-sided')

answer_1 = round(result[0], 3)
answer_2 = round(result[1], 4)
answer_3 = ['기각' if answer_2 < 0.05 else '채택']

print(answer_1, answer_2, answer_3)

#######################################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-1.csv')

train = df.iloc[0:int(len(df)*0.7),:]

before = round(np.nanmedian(train['Ozone']),1)

mean = round(np.nanmean(train['Ozone']),1)

train['Ozone'] = train['Ozone'].fillna(mean)

after = round(train['Ozone'].median(),1)

answer = round(abs(before - after),1)

print(answer)

# 7.7

#####################################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-2.csv')

train = df[(df['HAIR'] == 'White Hair') & (df['EYE'] == 'Blue Eyes')]

print(train.info())

mean = round(np.nanmean(train['APPEARANCES']), 2)
std = round(np.nanstd(train['APPEARANCES']), 2)

result = train[(train['APPEARANCES'] >= (mean-(1.5*std))) & (train['APPEARANCES'] <= (mean+(1.5*std)))]

answer = round(result['APPEARANCES'].mean(), 2)

print(answer)

# 30.15

#############################################################################################################

import pandas as pd
import numpy as np
from scipy.stats import iqr

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-3.csv')

mean = df['Sales'].mean()
std = df['Sales'].std()

train = df[(df['Sales'] >= (mean-(1.5*std))) & (df['Sales'] <= (mean+(1.5*std)))]

answer = round(train['Age'].std(), 2)

print(answer)

# 16.05

###############################################################################################################

import pandas as pd
import numpy as np
from scipy.stats import iqr

df = pd.read_csv('C:/Users/82108/Desktop/data/mtcars.csv')

train = df.iloc[0:int(len(df)*0.75), :]
valid = df.iloc[int(len(df)*0.75):, :]
print(len(train), len(valid), len(df))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x_train = train[['drat', 'wt', 'gear', 'carb']]
y_train = train['mpg']
x_train = pd.get_dummies(x_train)

x_valid = valid[['drat', 'wt', 'gear', 'carb']]
y_valid = valid['mpg']
x_valid = pd.get_dummies(x_valid)

model = LinearRegression()

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(round(mean_squared_error(y_valid, pred, squared = False),3))

############################################################################################################

import numpy as np
import pandas as pd
from scipy.stats import f

a = np.array([1,2,3,4,6])
b = np.array([4,5,6,7,8])

def f_test(a,b) :
    x = np.var(a, ddof = 1)
    y = np.var(b, ddof = 1)

    if x < y :
        x , y = y, x

    fvalue = x / y
    a_dof = len(a) -1
    b_dof = len(b) -1
    pvalue = (1-f.cdf(fvalue, a_dof, b_dof)) * 2

    return fvalue, pvalue

answer_1, answer_2 = f_test(a,b)

answer_1 = round(answer_1, 2)
answer_2 = round(answer_2, 4)
answer_3 = ['기각' if answer_2 <0.05 else '채택']

print(answer_1, answer_2, answer_3)

###########################################################################################################

import numpy as np
import pandas as pd

num = pd.DataFrame({'male' : [340], 'female' : [540]})
total = num.iloc[0].sum()

rate = pd.DataFrame({'male' : [0.35], 'female' : [0.65]})

from scipy.stats import chisquare

result = chisquare(num.iloc[0], (rate.iloc[0] * total))

answer_1 = round(result[0],5)
answer_2 = round(result[1], 5)
answer_3 = ['기각' if answer_2 <0.05 else '채택']

print(answer_1, answer_2, answer_3)