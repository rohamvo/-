import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210303.csv')

df.info()

print(df)

df = df.dropna(subset = ['country', 'year', 'new_sp'])

print(df)

x = df[['country', 'year', 'new_sp']]

x = x[x['year'] == 2000]

mean = round(x['new_sp'].mean(),2)

result = x[x['new_sp'] > mean]

print(len(result))

########################################################################

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/82108/Desktop/data/P220403.csv')

df.info()

print(df['date_added'].head(5))

df['date_added'] = pd.to_datetime(df['date_added'], format = '%B %d, %Y')

print(df['date_added'].head(5))

result = df[(df['country'] == 'United Kingdom') & (df['date_added'].dt.year == 2018) & (df['date_added'].dt.month == 1)]
print(len(result))

##########################################################################

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/82108/Desktop/data/P230601.csv')

df['call_time'] = pd.to_datetime(df['신고일시'], format = '%Y-%m-%d %H:%M')
df['depart_time'] = pd.to_datetime(df['출동일시'], format = '%Y-%m-%d %H:%M')

df['minus_time'] = (df['depart_time'] - df['call_time'])

df['minus_time'] = df['minus_time'].dt.total_seconds()

print(df['minus_time'])

result = df.groupby([df['출동소방서'], df['call_time'].dt.year, df['call_time'].dt.month])['minus_time'].mean()

result = pd.DataFrame(result)

print(result)

result = result.sort_values(by = 'minus_time', ascending = False)

a = round((result['minus_time'].iloc[0] / 60), 0)

print(int(a))

####################################################################################################

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/82108/Desktop/data/P230603.csv')

print(df.head())
print(df.info())
df['년월'] = pd.to_datetime(df['년월'])

print(df.info())

df['year'] = df['년월'].dt.year
df['month'] = df['년월'].dt.month
df['total'] = df['강력범'] + df['절도범'] + df['폭력범'] + df['지능범'] + df['풍속범'] + df['기타형사범']

result = df.groupby('year').mean('total')

result = result.sort_values(by = 'total', ascending = False)

print(int(result['total'].iloc[0]))