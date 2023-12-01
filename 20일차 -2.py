import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210201.csv')

df = df.sort_values(by = 'crim', ascending = False)

df['crim'].iloc[0:10] = df['crim'].iloc[9]

result = df[df['age'] >= 80]

answer = round(result['crim'].mean(), 2)

print(answer)

###############################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210202.csv')

train = df.iloc[0:int(len(df) * 0.8), :]

before = np.nanstd(train['total_bedrooms'])

median = np.nanmedian(train['total_bedrooms'])

train['total_bedrooms'] = train['total_bedrooms'].fillna(median)

after = train['total_bedrooms'].std()

answer = round(abs(before-after), 2)

print(answer)

##############################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210203.csv')

mean = df['charges'].mean()
std = df['charges'].std()

result = df[(df['charges'] <= (mean-(1.5*std))) | (df['charges'] >= (mean+(1.5*std)))]

answer = int(result['charges'].sum())

print(answer)

###############################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210301.csv')

df = df.dropna()

train = df.iloc[0:int(len(df) * 0.7), :]

q1 = np.quantile(train['housing_median_age'], q = 0.25)

answer = int(q1)

print(answer)

###############################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210302.csv')

result = df.isna().sum() / len(df)

result = pd.DataFrame(result)

result = result.sort_values(by = 0, ascending = False)

answer = result.index[0]

print(answer)

################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210303.csv')

df = df.dropna(subset = ['country', 'new_sp', 'year'])

train = df[['country', 'new_sp', 'year']]

train = train[train['year'] == 2000]

mean = round(train['new_sp'].mean(),2)

result = train[train['new_sp'] > mean]

answer = len(result)

print(answer)

################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220401.csv')

q1 = np.quantile(df['y'], q = 0.25)
q3 = np.quantile(df['y'], q = 0.75)

answer = int(q3-q1)

print(answer)

################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220402.csv')

df['rate'] = (df['num_loves'] + df['num_wows']) / df['num_reactions']

result = df[(df['rate'] > 0.4) & (df['rate'] < 0.5)]

answer = len(result)

print(answer)

################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220403.csv')
print(df['date_added'].head())

df['date_added'] = pd.to_datetime(df['date_added'], format = '%B %d, %Y')

print(df['date_added'].head())

result = df[(df['country'] == 'United Kingdom') & (df['date_added'].dt.year == 2018) & (df['date_added'].dt.month == 1)]

answer = len(result)

print(answer)

################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220501.csv', encoding = 'euc-kr')

result = df[(df['종량제봉투종류'] == '규격봉투') & (df['종량제봉투용도'] == '음식물쓰레기') & (df['2L가격'] != 0)]

answer = int(result['2L가격'].mean())

print(answer)

###################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220502.csv')

df['bmi'] = df['Weight'] / (df['Height'] / 100) ** 2

normal = len(df[(df['bmi'] >= 18.5)& (df['bmi'] <23)] )
danger = len(df[(df['bmi'] >= 23) & (df['bmi'] < 25)])

answer = abs(normal - danger)

print(answer)

###################################################################

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/82108/Desktop/data/P230601.csv')

df.columns = ['call_time', 'start_time', 'name']

df['call_time'] = pd.to_datetime(df['call_time'], format = '%Y-%m-%d %H:%M')
df['start_time'] = pd.to_datetime(df['start_time'], format = '%Y-%m-%d %H:%M')
df['diff_time'] = df['start_time'] - df['call_time']
df['diff_time'] = df['diff_time'].dt.total_seconds()
df['year'] = df['call_time'].dt.year
df['month'] = df['call_time'].dt.month
result = df.groupby([df['name'], df['call_time'].dt.year, df['call_time'].dt.month]).mean()
result = result.sort_values(by = 'diff_time', ascending = False)
answer = round((result['diff_time'].iloc[0] / 60), 0)
print(answer)

###################################################################

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/82108/Desktop/data/P230603.csv')

df.columns = ['ym', 'a', 'b', 'c', 'd', 'e', 'f']

df['total'] = df['a'] + df['b'] + df['c'] + df['d'] + df['e'] + df['f']

df['ym'] = pd.to_datetime(df['ym'], format = '%Y-%m')

result = df.groupby(df['ym'].dt.year).mean()

result = result.sort_values(by = 'total', ascending = False)

print(result)

answer = int(result['total'].iloc[0])

print(answer)