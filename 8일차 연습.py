import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220502.csv')

df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

normal = df[(df['BMI'] >= 18.5) & (df['BMI'] < 23)]
danger = df[(df['BMI'] >= 23) & (df['BMI'] < 25)]

answer = int(abs(len(normal)-len(danger)))
print(answer)

# 답 28

df = pd.read_csv('C:/Users/82108/Desktop/data/P220503.csv', encoding = 'euc-kr')

df['순전입학생수'] = df['전입학생수(계)'] - df['전출학생수(계)']

df = df.sort_values(by = '순전입학생수', ascending = False)

df.head()

answer = df['전체학생수(계)'].iloc[0]

print(answer)

# 답 956

df = pd.read_csv('C:/Users/82108/Desktop/data/P230601.csv')

from datetime import datetime

df['call'] = pd.to_datetime(df['신고일시'])
df['go'] = pd.to_datetime(df['출동일시'])

df['minus'] = (df['go'] - df['call']).dt.total_seconds()

print(df['minus'])

result = df.groupby([df['출동소방서'], df['call'].dt.year, df['call'].dt.month]).mean('minus')

result = result.sort_values(by = 'minus', ascending = False)

answer = round(result['minus'].iloc[0]/60, 0)

print(answer)

# 답 64

df = pd.read_csv('C:/Users/82108/Desktop/data/P230602.csv')

df['rate'] = (df['student_1'] + df['student_2'] + df['student_3'] + df['student_4'] + df['student_5'] + df['student_6'] / df['teacher'])

df = df.sort_values(by = 'rate', ascending = False)

answer = df['teacher'].iloc[0]

print(answer)

# 답 90

df = pd.read_csv('C:/Users/82108/Desktop/data/P230603.csv')

from datetime import datetime
from pandas import to_datetime

df.info()

print(df)
df['year'] = df['년월'].str[:4]
df['total'] = df['강력범'] + df['절도범'] + df['폭력범'] + df['지능범'] + df['풍속범'] + df['기타형사범']

result = df.groupby('year').sum('total')

result = result.sort_values(by = 'total', ascending = False)

print(result['total'])

answer = int(result['total'].iloc[0] / 12)

print(answer)

# 답 19329