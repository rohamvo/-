import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210201.csv')

df = df.sort_values(by = 'crim', ascending=False)

print(df['crim'])

print(df['crim'].head(20))

print(df['crim'].iloc[9])

df['crim'][0:10] = df['crim'].iloc[9]

result_df = df[df['age'] >= 80]
answer = round(result_df['crim'].mean(),2)
print(answer)

# 답 : 5.76

df = pd.read_csv('C:/Users/82108/Desktop/data/P210202.csv')

print(df)
print(df.iloc[0:int(len(df) * 0.8)])
before_df = df.iloc[0:int(len(df)*0.8)]
after_df = before_df['total_bedrooms'].fillna(before_df['total_bedrooms'].median())
before_std = before_df['total_bedrooms'].std()
print(after_df)
after_std = after_df.std()

answer = round(abs(before_std - after_std), 2)
print(answer)

# 답 1.98

df = pd.read_csv('C:/Users/82108/Desktop/data/P210203.csv')

mean = df['charges'].mean()
std = df['charges'].std()
result = df[df['charges'] >= mean + (1.5*std)]
answer = int(np.sum(result['charges']))
print(answer)

# 답 621430
from scipy.stats import iqr
df = pd.read_csv('C:/Users/82108/Desktop/data/P210301.csv')

df = df.dropna()

train_df = df.iloc[0:int(len(df) * 0.7)]

print(train_df)

train_df.info()

answer = int(np.quantile(train_df['housing_median_age'], q = 0.25))
print(answer)

# 답 : 19

df = pd.read_csv('C:/Users/82108/Desktop/data/P210303.csv')

len(df)

df = df[['country', 'year', 'new_sp']].dropna()

df_2000 = df[df['year'] == 2000]

mean = round(df_2000['new_sp'].mean(),2)

result = df_2000[df_2000['new_sp'] > mean]

answer = len(result)

print(answer)

# 답 38

import numpy as np
df = pd.read_csv('C:/Users/82108/Desktop/data/P220401.csv')

q1 = np.quantile(df['y'], q = 0.25)
q3 = np.quantile(df['y'], q = 0.75)

answer = int(q3-q1)
print(answer)

# 답 36

df = pd.read_csv('C:/Users/82108/Desktop/data/P220402.csv')

df['p_rate'] = (df['num_loves'] + df['num_wows']) / df['num_reactions']

result = df[(df['p_rate'] > 0.4) & (df['p_rate'] < 0.5)]
answer = len(result)
print(answer)

# 답 90

df = pd.read_csv('C:/Users/82108/Desktop/data/P220403.csv')

df['date_added'] = pd.to_datetime(df['date_added'], format = '%B %d, %Y')

result = df[(df['country'] == 'United Kingdom') & (df['date_added'].dt.year == 2018) & (df['date_added'].dt.month == 1)]

answer = len(result)

print(answer)

# 답 6

df = pd.read_csv('C:/Users/82108/Desktop/data/P220501.csv', encoding = 'euc-kr')

result = df[(df['종량제봉투종류'] == '규격봉투') & (df['종량제봉투용도'] == '음식물쓰레기') & (df['2L가격'] != 0)]

answer = round(np.mean(result['2L가격']), 0)

print(result['2L가격'].mean())

print(answer)
