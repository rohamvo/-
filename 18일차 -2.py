import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210201.csv')

df = df.sort_values(by = 'crim', ascending = False)

print(df['crim'].head(10))

df['crim'].iloc[0:10] = df['crim'].iloc[9]

print(df['crim'].head(11))

result = df[df['age'] >= 80]

answer = round(result['crim'].mean(),2)

print(answer)

#################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210202.csv')

train = df.iloc[0:int(len(df) * 0.8), :]

before = np.nanstd(train['total_bedrooms'])

train['total_bedrooms'] = train['total_bedrooms'].fillna(train['total_bedrooms'].median())

after = train['total_bedrooms'].std()

answer = round(abs(before - after), 2)

print(answer)

##############################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210203.csv')

mean = df['charges'].mean()
std = df['charges'].std()

result = df[(df['charges'] <= mean-(1.5*std)) | (df['charges'] >= mean+(1.5*std))]

answer = int(result['charges'].sum())

print(answer)

##########################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210301.csv')

df = df.dropna()

train = df.iloc[0:int(len(df) * 0.7), :]

answer = int(np.quantile(train['housing_median_age'], q = 0.25))

print(answer)

#####################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210302.csv')

result = df.isna().sum() / len(df)

result = pd.DataFrame(result)

result = result.sort_values(by = 0, ascending = False)

print(result.index[0])

######################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210303.csv')

train = df[['country', 'new_sp', 'year']]
train = train.dropna()

print(train.info())

train_2000 = train[train['year'] == 2000]

mean = round(train_2000['new_sp'].mean(),3)
answer = len(train_2000[train_2000['new_sp'] >= mean])

print(answer)

#####################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220401.csv')

q1 = np.quantile(df['y'], q = 0.25)
q3 = np.quantile(df['y'], q = 0.75)

answer = int(q3-q1)

print(answer)

####################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220403.csv')

df['date_added'] = pd.to_datetime(df['date_added'], format = '%B %d, %Y')

print(df.info())

answer = len(df[(df['date_added'].dt.year == 2018) & (df['date_added'].dt.month == 1) & (df['country'] == 'United Kingdom')])

print(answer)

######################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220501.csv', encoding = 'euc-kr')

result = df[(df['종량제봉투종류'] == '규격봉투') & (df['종량제봉투용도'] == '음식물쓰레기') & (df['2L가격'] != 0)]

answer = int(result['2L가격'].mean())

print(answer)

#######################################################################################


import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220502.csv')

df.columns = ['a', 'b', 'c', 'd']

print(df.info())

df['bmi'] = df['Weight'] / (df['Height']/100) ** 2

normal = len(df[(df['bmi'] >= 18.5) & (df['bmi'] < 23)])
danger = len(df[(df['bmi'] >= 23) & (df['bmi'] < 25)])

answer = int(abs(normal - danger))

print(answer)

######################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220503.csv', encoding = 'euc-kr')

print(df.info())

df['순전입'] = df['전입학생수(계)'] - df['전출학생수(계)']

df = df.sort_values(by = '순전입', ascending = False)

print(df['전체학생수(계)'].iloc[0])

####################################################################################


import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/82108/Desktop/data/P230601.csv')

df.columns = ['call_time', 'start_time', 'name']


df['call_time'] = pd.to_datetime(df['call_time'], format = '%Y-%m-%d %H:%M')
df['start_time'] = pd.to_datetime(df['start_time'], format = '%Y-%m-%d %H:%M')
df['year'] = df['call_time'].dt.year
df['month'] = df['call_time'].dt.month
df['minus'] = (df['start_time'] - df['call_time']).dt.total_seconds()

result = df.groupby([df['name'], df['year'], df['month']]).mean()

result = result.sort_values(by = 'minus', ascending = False)

answer = round(result['minus'].iloc[0] / 60, 0)

print(answer)

#####################################################################################

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/82108/Desktop/data/P230602.csv')

df['rate'] = (df['student_1'] + df['student_2'] + df['student_3'] + df['student_4'] + df['student_5'] + df['student_6']) / df['teacher']
df = df.sort_values(by = 'rate', ascending = False)

print(df.head())

answer = df['teacher'].iloc[0]

print(answer)

######################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P230603.csv')

df['년월'] = pd.to_datetime(df['년월'], format = '%Y-%m')
df.columns = ['년월', 'a', 'b', 'c', 'd', 'e', 'f']
df['year'] = df['년월'].dt.year
df['total'] = df['a'] + df['b'] + df['c'] + df['d'] + df['e'] + df['f']

result = df.groupby('year').mean()

result = result.sort_values(by = 'total', ascending = False)

answer = round(result['total'].iloc[0], 0)

print(int(answer))