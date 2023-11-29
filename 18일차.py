import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-1.csv')

print(df.info())

train = df.iloc[0:int(len(df) * 0.7), :]

train = train.sort_values(by = 'price', ascending = False)

result = train.iloc[0:5]

answer = int(result['depth'].median())

print(answer)

###########################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-2.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
print(df.info())
df = df.dropna(subset = 'TotalCharges')
print(df.info())

mean = df['TotalCharges'].mean()
std = df['TotalCharges'].std()

result = df[(df['TotalCharges'] > (mean-(1.5*std))) & (df['TotalCharges'] < (mean+(1.5*std)))]

answer = int(result['TotalCharges'].mean())

print(answer)

##########################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-3.csv')

am_1 = df[df['am'] == 1].sort_values(by = 'hp')
am_0 = df[df['am'] == 0].sort_values(by='hp')
am_1_5 = am_1.iloc[0:5]
am_0_5 = am_0.iloc[0:5]

result_1 = am_1_5['mpg'].mean()
result_0 = am_0_5['mpg'].mean()

answer = round((result_1 - result_0), 1)

print(answer)

#########################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-1.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-2.csv')

df1['Churn'] = df1['Churn'].map({"Yes" : 1, "No" : 0})
df1['Churn'] = df1['Churn'].astype('category')
df2['Churn'] = df2['Churn'].map({'Yes' : 1, "No" : 0})
df2['Churn'] = df2['Churn'].astype('category')
y = df1['Churn']
y_test = df2['Churn']
df1 = df1.drop('Churn', axis = 1)
df2 = df2.drop('Churn', axis = 1)
x = pd.get_dummies(df1)
x_test = pd.get_dummies(df2)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3)

model = RandomForestClassifier(n_estimators = 700)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))
print(accuracy_score(y_valid, pred))
print(precision_score(y_valid, pred))
print(f1_score(y_valid, pred))

pred_test = model.predict(x_test)

print(roc_auc_score(y_test, pred_test))
print(accuracy_score(y_test, pred_test))
print(precision_score(y_test, pred_test))
print(f1_score(y_test, pred_test))

pred_test = ['No' if p == 0 else 'Yes' for p in pred_test]

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

##################################################################################

import pandas as pd
import numpy as np

before = np.array([200,210,190,180,175])
after = np.array([180,175,160,150,160])

from scipy.stats import ttest_rel, shapiro

result = ttest_rel(after, before, alternative = 'less')

answer_1 = round(result.statistic, 5)
answer_2 = round(result.pvalue, 5)
answer_3 = ['기각' if result.pvalue < 0.05 else '채택']

print(answer_1, answer_2, answer_3)

################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-6.csv')

from scipy.stats import ttest_1samp, shapiro, wilcoxon

result = wilcoxon(df['Temp']-75, alternative = 'two-sided')

answer_1 = round(result.statistic, 3)
answer_2 = round(result.pvalue, 4)
answer_3 = ['기각' if result.pvalue < 0.05 else '채택']

print(answer_1, answer_2, answer_3)

################################################################################


import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-1.csv')

train = df.iloc[0:int(len(df) * 0.7), :]

before = round(np.nanmedian(train['Ozone']),1)

train['Ozone'] = train['Ozone'].fillna(round(np.mean(train['Ozone']), 1))

after = round(train['Ozone'].median())

answer = round((before - after),1)

print(answer)

################################################################################