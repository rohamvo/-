import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-1.csv')

train = df.iloc[0:int(len(df)*0.7),:]

before = round(np.nanmedian(train['Ozone']),1)

train['Ozone'] = train['Ozone'].fillna(round(np.nanmean(train['Ozone']),1))

after = round(np.median(train['Ozone']),1)

answer = round((before - after),1)

print(answer)

##############################################################################

import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-2.csv')

train = df[(df['HAIR'] == 'White Hair') & (df['EYE'] == "Blue Eyes")]

mean = round(np.nanmean(train['APPEARANCES']),2)
std = round(np.nanstd(train['APPEARANCES']),2)

result = train[(train['APPEARANCES'] >= (mean-(1.5*std))) & (train['APPEARANCES'] <= (mean + (1.5*std)))]

answer = round(result['APPEARANCES'].mean(), 2)

print(answer)

################################################################################

import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-3.csv')

mean = df['Sales'].mean()
std = df['Sales'].std()

train = df[(df['Sales'] >= (mean-(1.5*std))) & (df['Sales'] <= (mean+(1.5*std)))]

answer = round(train['Age'].std(), 2)

print(answer)

#####################################################################################

import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/82108/Desktop/data/mtcars.csv')

print(len(df)*0.75, len(df)*0.25)

train = df.iloc[0:int(len(df)*0.75), :]
valid = df.iloc[int(len(df)*0.75):, :]

print(len(train), len(valid))

y_train = train['mpg']
x_train = pd.get_dummies(train[['drat', 'wt', 'gear', 'carb']])

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()

model.fit(x_train, y_train)

y_valid = valid['mpg']
x_valid = pd.get_dummies(valid[['drat', 'wt', 'gear', 'carb']])

pred = model.predict(x_valid)

result = mean_squared_error(y_valid, pred, squared = False)

answer = round(result, 3)

print(answer)

###########################################################################

import numpy as np
import pandas as pd
from scipy.stats import f

a = np.array([1,2,3,4,6])
b = np.array([4,5,6,7,8])

def f_test(a,b) :
    x = np.var(a, ddof = 1)
    y = np.var(b, ddof = 1)

    if x < y :
        x, y = y, x
    fvalue = x / y
    a_dof = len(a) - 1
    b_dof = len(b) - 1
    pvalue = (1-f.cdf(fvalue, a_dof, b_dof)) * 2

    return fvalue, pvalue

fvalue, pvalue = f_test(a,b)

print(round(fvalue, 2), round(pvalue, 4))

answer_3 = ['기각' if pvalue < 0.05 else '채택']

print(answer_3)

########################################################################################

import numpy as np
import pandas as pd

num = pd.DataFrame({'male' : [340], 'female' : [540]})
rate = pd.DataFrame({'male' : [0.35], 'female' : [0.65]})

from scipy.stats import chisquare

result = chisquare(num.iloc[0], rate.iloc[0] * num.iloc[0].sum())

print(result)

answer_1 = round(result.statistic, 5)
answer_2 = round(result.pvalue, 5)
answer_3 = ['기각' if result.pvalue < 0.05 else '채택']

print(answer_1, answer_2, answer_3)