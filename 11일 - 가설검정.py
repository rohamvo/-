# 추출된 표본이 동일 모집단에 속하는지 검정 - z검정
# 모집단 분산을 이미 알고 있을 때 분포의 평균을 테스트 한다
# (표본평균 - 모평균) / (모표준편차 / 루트(표본크기))

import pandas as pd
import numpy as np
from scipy.stats import norm

a = np.array([25,27,31,23,24,30,26])

z = (np.mean(a) - 26) / (5/np.sqrt(len(a)))

pvalue = (1-norm.cdf(z)) * 2

print(pvalue)

###################################################

# 한 집단의 평균이 모집단의 평균과 같은지 검정 - t검정


import numpy as np

from scipy.stats import ttest_1samp, shapiro

a = np.array([12,14,16,19,11,17,13])

print(shapiro(a))

result = ttest_1samp(a, popmean = 11)

print(result)

#####################################################

# 정규성 만족하지 않는 검정 = wilcoxon
# 기준 평균 값 빼기 필요

import numpy as np
import pandas as pd

from scipy.stats import shapiro, wilcoxon

df = pd.read_csv('C:/Users/82108/Desktop/data/cats.csv')

print(shapiro(df['Bwt']))

result = wilcoxon(df['Bwt'] - 2.1, alternative = 'two-sided')
print(result)

#####################################################

# 쌍체표본 검정 = 전 후 비교
# 정규성 만족 필요
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

data = pd.DataFrame({'before' : [5,3,8,4,3,2,1], 'after' : [8,6,6,5,8,7,3]})

print(shapiro(data['before']))

print(shapiro(data['after']))

print(ttest_rel(data['before'], data['after'], alternative = 'less'))

#######################################################

# 서로 다른 모집단에서 추출된경우
# 두 집단의 평균 차이 검정 - 독립표본 t검정
# 정규성, 등분산성 만족 필요
# 표본 10개 미만 - 만 위트니u검정(Mann Whitneyu)
# 10 ~ 30 - 샤피로
# 30개 이상 - 중심 극한정리로 정규성 만족

import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind

df = pd.read_csv('C:/Users/82108/Desktop/data/cats.csv')

g1 = df[df['Sex'] == 'F']['Bwt']
g2 = df[df['Sex'] == 'M']['Bwt']

print(levene(g1,g2))

print(ttest_ind(g1,g2,equal_var = False))

######################################################

# f-검정은 두 표본의 분산에 대한 차이가 통계적으로 유의한지 판별하는 검정 기법

from scipy.stats import f
import numpy as np

a = np.array([1,2,3,4,6])
b = np.array([4,5,6,7,8])

def f_test(a,b) : 
    x = np.var(a, ddof = 1)
    y = np.var(b, ddof = 1)

    if x < y :
        x, y = y, x

    a_dof = len(a) - 1
    b_dof = len(b) - 1
    fvalue = x/y
    pvalue = (1-f.cdf(fvalue, a_dof, b_dof)) *2

    return fvalue, pvalue

print(f_test(a,b))

########################################################
# 카이제곱 검정
# 적합도검정
# 표본 집단의 분포가 주어진 특정 분포를 따르고 있는지를 검정하는 기법
# 적합도 검정 자유도 = 범주의 수 -1

import numpy as np
from scipy.stats import chisquare

num = np.array([90,160])
exp = np.array([0.45, 0.55]) * np.sum(num)

print(chisquare(num, exp))

########################################################

# 독립성 검정
# 각 범주가 서로 독립적인지, 서로 연관성이 있는지를 검정하는 기법
# 자유도 = (범주 1의 수 -1) * (범주 2의 수 -1)
# crosstab(범주 1, 범주2)

import pandas as pd
from scipy.stats import chi2_contingency

survey = pd.read_csv('C:/Users/82108/Desktop/data/survey.csv')

tb = pd.crosstab(survey['Sex'], survey['Exer'])

print(chi2_contingency(tb))


