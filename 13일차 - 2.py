import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-1.csv')

train = df.iloc[0:int(len(df) * 0.7), :]

before = np.nanmedian(train['Ozone'])

train['Ozone'] = train['Ozone'].fillna(np.nanmean(train['Ozone']))

after = np.median(train['Ozone'])

answer = round((before-after), 1)

print(answer)

# 답 -7.7

###################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-2.csv')

print(df.info())

df.dropna(subset = ['EYE', 'HAIR'])

print(df.info())

train = df[(df['HAIR'] == 'White Hair') & (df['EYE'] == 'Blue Eyes')]

mean = round(train['APPEARANCES'].mean(), 2)
std = round(train['APPEARANCES'].std(), 2)

result = train[(train['APPEARANCES'] >= (mean-(1.5*std))) & (train['APPEARANCES'] <= (mean+(1.5*std)))]

answer = round(result['APPEARANCES'].mean(), 2)

print(answer)

# 답 30.15

#######################################################################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M2-3.csv')

df.info()

mean = df['Sales'].mean()
std = df['Sales'].std()

train = df[(df['Sales'] >= (mean-(1.5*std))) & (df['Sales'] <= (mean+(1.5*std)))]

answer = round(train['Age'].std(), 2)

print(answer)

# 답 16.05

###################################################################################################


import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/mtcars.csv')

train = df.iloc[0:int(len(df) * 0.75), :]
test = df.iloc[int(len(df)*0.75):, :]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

x_train = pd.get_dummies(train[['drat', 'wt', 'gear', 'carb']])
y_train = train['mpg']
x_valid = pd.get_dummies(test[['drat', 'wt', 'gear', 'carb']])
y_valid = test['mpg']

model = LinearRegression()

model.fit(x_train, y_train)

pred = model.predict(x_valid)

answer = round(mean_squared_error(y_valid, pred, squared = False),3)

print(answer)

#############################################################################

import pandas as pd
import numpy as np
from scipy.stats import f

a = np.array([1,2,3,4,6])
b = np.array([4,5,6,7,8])

def f_test(a,b) :
    x = np.var(a, ddof = 1)
    y = np.var(b, ddof = 1)

    if x < y :
        x, y  = y, x
    a_dof = len(a) - 1
    b_dof = len(b) - 1
    fvalue = round((x / y), 2)
    pvalue = round((1-f.cdf(fvalue, a_dof, b_dof))*2,4)

    return fvalue, pvalue

print(f_test(a,b))

# 1.48, 0.7133, 귀무가설 채택

############################################################################

import numpy as np
import pandas as pd
from scipy.stats import chisquare, shapiro

p = pd.DataFrame({"male" : [340], "female" : [540]})
r = pd.DataFrame({'male' : [0.35], 'female' : [0.65]})

p_l = np.array(list(p.iloc[0]))
r_l = np.array(list(r.iloc[0]))

result = chisquare(p_l, (r_l * (p_l.sum())))

answer_1 = round(result.statistic, 5)
answer_2 = round(result.pvalue, 5)

print(answer_1, answer_2)

# 5.11489, 0.02372, 귀무가설 기각
