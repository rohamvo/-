import pandas as pd
from scipy.stats import chisquare

df = pd.read_csv('C:/Users/82108/Desktop/data/P230605.csv', encoding='euc-kr')

# 문제 1번

answer_1 = round((len(df[df['코드'] == 4]) / len(df)), 3)
print(answer_1)

# 문제 2,3번

total_cnt = len(df)
count_df = df.groupby('코드').size()
rate_df = pd.DataFrame({'코드' : [1,2,3,4], '비율' : [0.05, 0.1, 0.05, 0.8]})
rate_df['count'] = rate_df['비율'] * total_cnt

chis = chisquare(count_df, rate_df['count'])
print(chis)

answer_2 = round(chis.statistic, 3)
print(answer_2)

answer_3 = round(chis.pvalue, 3)
print(answer_3)

###################################################################

import pandas as pd

df = pd.read_csv('C:/Users/82108/Desktop/data/P230606.csv')

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

x = df[['O3', 'Solar', 'Wind']]
y = df['Temperature']

lm = LinearRegression()

lm.fit(x,y)

coefs = pd.DataFrame({'feature' : ['O3', 'Solar', 'Wind'], 'coefficient' : lm.coef_})

print(coefs)

answer_1 = round(float(coefs['coefficient'].iloc[0]),3)
print(answer_1)

tt = stats.ttest_ind(x['Wind'], y)
print(tt.pvalue)
answer_2 = round(tt.pvalue,3)
print(answer_2)

test = pd.DataFrame({'O3' : [10], 'Solar' : [90], 'Wind' : [20]})
pred = lm.predict(test)
print(pred)
answer_3 = np.round(pred, 3)
print(answer_3)
print(type(pred))