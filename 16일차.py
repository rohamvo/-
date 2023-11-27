import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-1.csv')

print(df.info())

train = df.iloc[0:int(len(df)*0.7), :]

train = train.sort_values(by = 'price', ascending = False)

result = train.iloc[0:5]

print(len(result))

answer = int(result['depth'].median())

print(answer)

####################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-2.csv')

df['TotalCharges'] = df['TotalCharges'].str.replace(' ', '')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['TotalCharges'] = df['TotalCharges'].dropna()

mean = df['TotalCharges'].mean()
std = df['TotalCharges'].std()

result = df[(df['TotalCharges'] > (mean-(1.5*std))) & (df['TotalCharges'] < (mean+(1.5*std)))]

answer = int(result['TotalCharges'].mean())

print(answer)

###################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-3.csv')

am_1 = df[df['am'] == 1]
am_0 = df[df['am'] == 0]

am_1 = am_1.sort_values(by = 'hp')
am_0 = am_0.sort_values(by = 'hp')

am_1_s5 = am_1.iloc[0:5]
am_0_s5 = am_0.iloc[0:5]

am_1_mean = am_1_s5['mpg'].mean()
am_0_mean = am_0_s5['mpg'].mean()

answer = round(am_1_mean - am_0_mean, 1)

print(answer)

####################################################################

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-1.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-2.csv')

print(df1.info())
print(df2.info())

df1['Churn'] = df1['Churn'].map({'Yes' : 1, "No" : 0})
y = df1['Churn']
df1 = df1.drop('Churn', axis = 1)
x = pd.get_dummies(df1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))
print(f1_score(y_valid, pred))
print(accuracy_score(y_valid, pred))

df2['Churn'] = df2['Churn'].map({'Yes' : 1, "No" : 0})
y_test = df2['Churn']
df2 = df2.drop('Churn', axis = 1)
x_test = pd.get_dummies(df2)

pred_test = model.predict(x_test)

print(roc_auc_score(y_test, pred_test))
print(f1_score(y_test, pred_test))
print(accuracy_score(y_test, pred_test))

pred_test = ['No' if p == 0 else 'Yes' for p in pred_test]

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

##############################################################################

import pandas as pd
import numpy as np

before = np.array([200,210,190,180,175])
after = np.array([180,175,160,150,160])

from scipy.stats import ttest_rel, shapiro, wilcoxon

print(shapiro(before))
print(shapiro(after))

result = ttest_rel(after, before, alternative = 'less')

answer_1 = round(result.statistic, 5)
answer_2 = round(result.pvalue, 5)
answer_3 = ['기각' if answer_2 < 0.05 else '채택']
print(answer_1, answer_2, answer_3)

###############################################################################

import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-6.csv')

from scipy.stats import wilcoxon, shapiro, ttest_1samp

print(df.info())

print(shapiro(df['Temp']))

result = wilcoxon(df['Temp'] - 75, alternative = 'two-sided')

answer_1 = round(result.statistic, 3)
answer_2 = round(result.pvalue, 4)
answer_3 = ['기각' if answer_2 < 0.05 else '채택']

print(answer_1, answer_2, answer_3)