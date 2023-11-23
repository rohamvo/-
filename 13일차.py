import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-1.csv')

train = df.iloc[0:int(len(df) * 0.7), :]
train = train.sort_values(by = 'price', ascending = False)
result = train.iloc[0:5]

answer_1 = int(result['depth'].median())
print(answer_1)

# 답 62

#######################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-2.csv')

print(df.info())

df.dropna(subset = ['TotalCharges'])

print(df.info())
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.replace(' ', ''))
std = df['TotalCharges'].std()
mean = df['TotalCharges'].mean()

result = df[(df['TotalCharges'] >= (mean-(1.5*std))) & (df['TotalCharges'] <= (mean+(1.5*std)))]

answer = int(result['TotalCharges'].mean())

print(answer)

# 답 1663

#####################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-3.csv')

df1 = df[df['am'] == 1]
df1 = df1.sort_values(by = 'hp')
df0 = df[df['am'] == 0]
df0 = df0.sort_values(by = 'hp')

df1 = df1.iloc[0:5]
print(df1)
df0 = df0.iloc[0:5]
print(df0)

print(df1['mpg'].mean())
print(df0['mpg'].mean())

answer = round((df1['mpg'].mean() - df0['mpg'].mean()), 1)

print(answer)

# 답 8.4

######################################################################


import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-1.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-2.csv')

#df1.info()
#df2.info()

df1['Churn'] = df1['Churn'].astype('category')
y = df1['Churn']
df1 = df1.drop('Churn', axis = 1)
x = pd.get_dummies(df1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(accuracy_score(y_valid, pred))

df2['Churn'] = df2['Churn'].astype('category')
df2_result = df2['Churn']
df2 = df2.drop('Churn', axis = 1)
x_test = pd.get_dummies(df2)

pred_test = model.predict(x_test)

accuracy_score(df2_result, pred_test)

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('result.csv', index = False)

a = pd.read_csv('result.csv')

print(a)

####################################################################

import numpy as np
import pandas as pd

before = np.array([200,210,190,180,175])
after = np.array([180,175,160,150,160])

from scipy.stats import shapiro, ttest_rel

print(shapiro(before), shapiro(after))

result = ttest_rel(after, before, alternative = 'less')

answer_1 = round(result.statistic, 5)
answer_2 = round(result.pvalue, 5)

print(answer_1, answer_2)

# 귀무가설 기각, 대립가설 채택

######################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-6.csv')

from scipy.stats import shapiro, ttest_1samp, wilcoxon

print(shapiro(df['Temp'])) # 정규성 만족 x

result = wilcoxon(df['Temp']-75, alternative = 'two-sided')

answer_1 = round(result.statistic, 3)
answer_2 = round(result.pvalue, 4)

print(result)

print(answer_1, answer_2)

# 귀무가설 기각, 대립가설 채택
