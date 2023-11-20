# 1.

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-1.csv')

train = df.iloc[0:int(len(df)*0.7)]
train = train.sort_values(by = 'price', ascending = False)

result = train.iloc[:5]

print(result)

answer_1 = int(np.median(result['depth']))
print(answer_1)

# 답 62

###################################################################

# 2.
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-2.csv')

df.info()

df['TotalCharges'].dropna()

df.info()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce') # pd.to_numeric 몰랐음
df.info()
mean = np.mean(df['TotalCharges'])
std = np.std(df['TotalCharges'])
print(mean, std)
result = df[(df['TotalCharges'] >= mean - (1.5*std)) & (df['TotalCharges'] <= mean + (1.5*std))]
result.dropna()

answer_2 = int(np.mean(result['TotalCharges']))
print(answer_2)

# 답 1663

###################################################################

# 3.

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-3.csv')

result_1 = df[df['am'] == 1]
result_1 = result_1.sort_values(by = 'hp').iloc[0:5]
print(len(result_1))
mean_1 = np.mean(result_1['mpg'])

result_2 = df[df['am'] == 0]
result_2 = result_2.sort_values(by = 'hp').iloc[0:5]
print(len(result_2))
mean_2 = np.mean(result_2['mpg'])

answer_3 = round(abs(mean_1 - mean_2), 1)
print(answer_3)

# 답 8.4

##################################################################

# 4.

import pandas as pd
import numpy as np

df1 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-1.csv')
df2 = pd.read_csv('C:/Users/82108/Desktop/data/M1-4-2.csv')

df1.info()
df2.info()

df1['Churn'] = df1['Churn'].astype('category') # 원래는 모든 범주형 컬럼 전부다 바꾸려고 했었음
y = df1['Churn']
df1 = df1.drop('Churn', axis = 1)
x = pd.get_dummies(df1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train,y_train)

pred_train = model.predict(x_valid)

print(accuracy_score(y_valid, pred_train))

df2 = df2.drop('Churn', axis = 1)

x_test = pd.get_dummies(df2)

pred_test = model.predict(x_test)

print(pred_test)

result = pd.DataFrame({'pred' : pred_test})

result.to_csv('수험번호.csv', index = False)

a = pd.read_csv('수험번호.csv')

print(a)

######################################################################

# 5.

import numpy as np
from scipy.stats import ttest_rel

before = np.array([200,210,190,180,175])
after = np.array([180, 175, 160, 150, 160])

result = ttest_rel(after, before, alternative = 'less') # after, before 순서 주의하자 문제에서 주어진대로 하자!


print(result)

answer_1 = round(result.statistic, 5)
answer_2 = round(result.pvalue, 5)

print(answer_1)
print(answer_2)

# 기각(귀무가설 채택) # 귀무가설 기준인것으로 보임 기각 채택 선택도 코딩으로 준비하자

if answer_2 < 0.05 : 
    print('기각')
else : 
    print('채택')

########################################################################

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_1samp, wilcoxon

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-6.csv')

df.info()

temp = df['Temp']

print(shapiro(temp))

result = wilcoxon(df['Temp']-75, alternative='two-sided')

answer_1 = round(result.statistic, 3)
answer_2 = round(result.pvalue, 4)

print(answer_1)
print(answer_2)

# 채택(대립가설 채택)