# 1.

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-1.csv')

train = df.iloc[0:int(len(df) * 0.7),:]

df.info()

o_mean = round(np.nanmean(df['Ozone']),1)
print(o_mean)

before = np.nanmedian(train['Ozone'])  ## nanmean 써야함 주의하자
print(before)

train['Ozone'] = train['Ozone'].fillna(o_mean)

# train.loc[train['Ozone'].isna(), 'Ozone'] = o_mean

after = np.median(train['Ozone'])
print(after)

answer = round(abs(before-after),1)
print(answer)

##################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-2.csv')


df.info()

train = df[(df['HAIR'] == 'White Hair') & (df['EYE'] == 'Blue Eyes')]

mean = round(np.mean(train['APPEARANCES']), 2)
std = round(np.std(train['APPEARANCES']), 2)

result = df[(df['APPEARANCES'] > (mean-std)) & (df['APPEARANCES'] < (mean+std))]

answer = round(result['APPEARANCES'].mean(), 2)
print(answer)

# 답 11.55

##################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-3.csv')

df.info()

mean = np.mean(df['Sales'])
print(mean)
std = np.std(df['Sales'])
print(std)

train = df[(df['Sales'] > (mean-std)) & (df['Sales'] < (mean+std))]

answer = round(np.std(train['Age']), 2)
print(answer)

# 답 16.14

################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/mtcars.csv')

df.info()

train_df = df.iloc[0:int(len(df) * 0.75),:]
test_df = df.iloc[int(len(df) * 0.75) + 1 :, :]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x = pd.get_dummies(train_df[['dart', 'wt', 'gear', 'carb']])
y = train_df['mpg']

model = LinearRegression()
model.fit(x,y)


###################################################################################

import numpy as np
from scipy.stats import f

a = np.array([1,2,3,4,6])
b = np.array([4,5,6,7,8])

def f_test(a,b) :
    x = np.var(a, ddof = 1)
    y = np.var(b, ddof=1)
    if x < y :
        x, y = y, x
    f_value = x/y
    x_dof = a.size -1
    y_dof = b.size -1
    p_value = round((1-f.cdf(f_value, x_dof, y_dof)) *2, 4)

    if p_value < 0.05 :
        result = '기각'
    else :
        result = '채택'

    return f_value, p_value, result

result = f_test(a, b)
print(result)

##############################################################################
import pandas as pd
import numpy as np
from scipy.stats import chisquare

n = pd.DataFrame({'Male' : [340], 'Female' : [540]})
rate = pd.DataFrame({'Male' : [0.35], 'Female' : [0.65]})

print(n.values[0])

result = chisquare(n.values.tolist()[0], (rate.values.tolist()[0] * (np.sum(n.values.tolist()[0]))))
print(result)

print(round(result.statistic), 5)
print(round(result.pvalue), 5)
if result.pvalue < 0.05 :
    print('기각')
else :
    print('채택')              


###########################################################################    




