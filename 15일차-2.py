import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210201.csv')

df = df.sort_values(by = 'crim', ascending = False)
df['crim'].iloc[0:10] = df['crim'].iloc[9]

print(df['crim'].head(11))

result = df[df['age']>=80]

print(round(result['crim'].mean(), 2))

#########################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210202.csv')

train = df.iloc[0:int(len(df)*0.8),:]

print(train)

before = df['total_bedrooms'].std()

df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

after = df['total_bedrooms'].std()

print(round(np.abs(before-after), 2))

#########################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210203.csv')

print(df.info())

mean = df['charges'].mean()
std = df['charges'].std()

result = df[(df['charges'] <= (mean-(1.5*std))) | (df['charges'] >= (mean + (1.5*std)))]

print(int(result['charges'].sum()))

#########################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210301.csv')

print(df.info())

df = df.dropna()

print(df.info())

train = df.iloc[0:int(len(df) * 0.7), :]
print(int(np.quantile(train['housing_median_age'], q = 0.25)))

##########################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210302.csv')

result = df.isna().sum() / len(df)

result = result.sort_values(ascending = False)

result = pd.DataFrame(result)

print(result[0].index[0])

##########################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220401.csv')

one = np.quantile(df['y'], q = 0.25)
three = np.quantile(df['y'], q = 0.75)
print(int(three-one))

##########################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220402.csv')

df['rate'] = (df['num_loves'] + df['num_wows']) / df['num_reactions']

result = df[(df['rate'] > 0.4) & (df['rate'] < 0.5)]

print(len(result))

##########################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P220502.csv')

df['bmi'] = df['Weight'] / (df['Height'] / 100) ** 2

