import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/82108/Desktop/data/P230605.csv', encoding = 'euc-kr')

from scipy.stats import chisquare

answer_1 = round((len(df[df['코드'] == 4]) / len(df)), 3)

rate = np.array([0.05, 0.1, 0.05, 0.8])

count = df.groupby('코드').size()

result = chisquare(count, (rate * len(df)))

answer_2 = round(result.statistic, 3)
answer_3 = round(result.pvalue, 3)

print(answer_1, answer_2, answer_3)

######################################################################################

import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/82108/Desktop/data/P230606.csv')

from sklearn.linear_model import LinearRegression
from scipy import stats

x = df[['O3', 'Solar', 'Wind']]
y = df['Temperature']
lm = LinearRegression()
lm.fit(x,y)

coefs = pd.DataFrame({'Feature' : ['O3', 'Solar', 'Wind'], 'Coef' : lm.coef_})


answer_1 = round(coefs[coefs['Feature'] == 'O3']['Coef'][0], 3)
print(answer_1)

from scipy.stats import ttest_ind

result = ttest_ind(df['Wind'], df['Temperature'])

answer_2 = round(result.pvalue, 3)

x_test = pd.DataFrame({'O3' : [10], 'Solar' : [90], 'Wind' : [20]})

pred = lm.predict(x_test)

answer_3 = round(pred[0], 3)

print(answer_1, answer_2, answer_3)