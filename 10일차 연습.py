from scipy.stats import norm
import numpy as np

x = np.array([25,27,31,23,24,30,26])

mean = np.mean(x)

z = (mean-26) / (5 / np.sqrt(7))

print(z)

p = (1- norm.cdf(z)) *2

print(p)

##################################################

from scipy.stats import shapiro, ttest_1samp
import numpy as np

new = np.array([12,14,16,19,11,17,13])

print(shapiro(new)) # pvalue가 0.853으로 유의수준 0.05 보다 크기때문에 정규성을 만족한다

print(ttest_1samp(new, popmean = 11)) # pvalue 가 0.016으로 유의수준 0.05 보다 작기때문에 대립가설 채택

###################################################

import pandas as pd
import numpy as np
from scipy.stats import shapiro, wilcoxon

df = pd.read_csv('C:/Users/82108/Desktop/data/cats.csv')

print(df.info())

print(shapiro(df['Bwt'])) # pvalue 가 유의수준 0.05보다 작기 때문에 대립가설(정규성 만족하지 않음) 채택

# 정규성을 만족하지 않음으로 윌콕슨 사용

print(wilcoxon(df['Bwt'] - 2.1, alternative = 'two-sided').statistic)

#############################################################

import pandas as pd
from scipy.stats import ttest_rel, shapiro

data = pd.DataFrame({'before' : [5,3,8,4,3,2,1], 'after' : [8,6,6,5,8,7,3]})

print(shapiro(data['before']), shapiro(data['after']))

result = ttest_rel(data['before'], data['after'], alternative = 'less')

print(result)

#################################################################

import pandas as pd
from scipy.stats import ttest_ind, shapiro, levene

df = pd.read_csv('C:/Users/82108/Desktop/data/cats.csv')

female = df[df['Sex'] == 'F']['Bwt']
male = df[df['Sex'] == 'M']['Bwt']

print(levene(female, male))

result = ttest_ind(female, male, equal_var = False)

print(result)

##########################################################

import numpy as np
from scipy.stats import f

df1 = np.array([1,2,3,4,6])
df2 = np.array([4,5,6,7,8])

def f_test(x,y) : 
    if np.var(x, ddof=1) < np.var(y, ddof=1) :
        x, y = y, x
    f_value = np.var(x, ddof=1) / np.var(y, ddof = 1)
    x_dof = x.size -1
    y_dof = y.size -1

    p_value = (1- f.cdf(f_value, x_dof, y_dof)) * 2
    return f_value, p_value

result = f_test(df1, df2)
print(result)

#########################################################

import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv('C:/Users/82108/Desktop/data/survey.csv')

test = pd.crosstab(df['Sex'], df['Exer'])
print(test)

result = chi2_contingency(test)
print(result)
print(result.pvalue)