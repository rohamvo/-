import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210201.csv')

df = df.sort_values(by = 'crim', ascending = False)

df['crim'].iloc[0:10] = df['crim'].iloc[9]

print(df['crim'].head(10))

answer = round(np.mean(df[df['age'] >= 80]['crim']),2)
print(answer)

###############################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210202.csv')

print(df.info())

train = df.iloc[0:int(len(df) * 0.8), :]
before = np.std(train['total_bedrooms'])
print(before)
median = np.nanmedian(train['total_bedrooms'])
median_test = train['total_bedrooms'].median()
print(median)
print(median_test)
# train['total_bedrooms'] = train['total_bedrooms'].fillna(np.median(train['total_bedrooms']))
train.loc[train['total_bedrooms'].isna() == True, :] = median 
after = np.std(train['total_bedrooms'])
print(after)

result = round(abs(before-after),2)

print(result)

#############################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210203.csv')

df.info()

mean = df['charges'].mean()
print(mean)
std = df['charges'].std()
print(std)

result = df[(df['charges'] <= (mean-(1.5*std))) | (df['charges'] >= (mean+(1.5*std)))]

print(result)

answer = int(result['charges'].sum())

print(answer)

############################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210301.csv')

df.info()

df.dropna()

train = df.iloc[0:int(len(df) * 0.7), :]

answer = int(np.quantile(train['housing_median_age'], q = 0.25))

print(answer)

###########################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210302.csv')

df.info()

result = df.isna().sum() / len(df)

result = pd.DataFrame(result)

result = result.sort_values(by = 0, ascending = False)

print(result)

answer = result.index[0]

print(answer)

###############################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P210303.csv')

df[['country', 'year', 'new_sp']].dropna()

print(df.head())

print(df.info())

#############################################################

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/82108/Desktop/data/P230603.csv')

print(df)

df['년월'] = datetime.strptime(df['년월'], format = '%b-%y')

print(df['년월'])