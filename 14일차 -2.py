import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/M1-6.csv')

from scipy.stats import wilcoxon, shapiro

print(shapiro(df['Temp']))

df.info()

result = wilcoxon(df['Temp'] - 75, alternative = 'two-sided')

print(result.statistic, result.pvalue)

result_2 = wilcoxon(df['Temp'] -75, alternative = 'two-sided', zero_method= 'wilcox', correction = False)

print(result_2.statistic, result_2.pvalue)

##############################################################

import pandas as pd
import numpy as np

before = np.array([200, 210, 190, 180, 175])
after = np.array([180,175,160,150,160])

from scipy.stats import wilcoxon, shapiro, ttest_rel

print(shapiro(before), shapiro(after))

result = ttest_rel(after, before,  alternative = 'less')

print(result.pvalue, result.statistic)

###############################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P230605.csv', encoding = 'euc-kr')

from scipy.stats import shapiro, chisquare

rate = pd.DataFrame({"1" : [0.05], "2" : [0.1], "3" : [0.05], "4" : 0.8})

answer_1 = round(((len(df[df['코드'] == 4])) / len(df)), 3)
# result = chisquare(df.groubpy['코드'].size(), )

result = chisquare(df.groupby('코드').size(), rate.iloc[0] * len(df))
print(result)

answer_2 = round(result.statistic, 3)
answer_3 = round(result.pvalue, 3)

print(answer_1, answer_2, answer_3)

###############################################################

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/82108/Desktop/data/P230606.csv')

print(df.info())

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

y = df['Temperature']
x = pd.get_dummies(df.drop('Temperature', axis = 1))

lm.fit(x, y)

print(round(lm.coef_[0],3))

from scipy.stats import ttest_ind

result = ttest_ind(df['Wind'], df['Temperature'])

print(round(result.pvalue, 3))

test = pd.DataFrame({'O3' : [10], 'Solar' : [90], 'Wind' : [20]})
x_test = pd.get_dummies(test)

pred = lm.predict(x_test)

print(round(pred[0], 3))

#######################################################################

import pandas as pd
import numpy as np

a = np.array([1,2,3,4,6])
b = np.array([4,5,6,7,8])

from scipy.stats import f

def f_test(a,b) :
    x = np.var(a, ddof = 1)
    y = np.var(b, ddof = 1)
    if x < y :
        x, y = y, x
    a_dof = len(a) -1
    b_dof = len(b) -1
    fvalue = x/y
    pvalue = (1-f.cdf(fvalue, a_dof, b_dof)) * 2

    return fvalue, pvalue

fvalue, pvalue = f_test(a,b)

print(round(fvalue, 2))
print(round(pvalue, 4))

# 귀무가설 채택

##########################################################################

import numpy as np
import pandas as pd

num = pd.DataFrame({'Male' : [340], 'Female' : [540]})
rate = pd.DataFrame({'Male' : [0.35], 'Female' : [0.65]})

from scipy.stats import chisquare

total_cnt = np.sum(num.iloc[0])
print(total_cnt)

result = chisquare(num.iloc[0], rate.iloc[0] * total_cnt)

print(result)

print(round(result.statistic, 5))
print(round(result.pvalue, 5))
# 귀무가설 기각