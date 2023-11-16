import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# 오존농도 변수에 대한 회귀계수 추정값을 출력하시오(반올림하여 소수점 셋째자리로)
# 답 : 0.172

# 오존농도, 일사량이 고정일때 풍속이 증가함에 따라 온도가 낮아진다는것을 검증했다. t-검증 값의 유의확률(p-value)
# 을 출력하시오. (반올림하여 소수점 셋째 자리로 출력)
# 답 : 0.0

# 어떤 날이 오존농도10, 일사량90, 풍속 20일 때 온도의 예측값을 출력하시오(반올맇마여 소수점 셋째 자리 출력)
# 답 : 68.334

df = pd.read_csv('C:/Users/82108/Desktop/data/P230606.csv')

print(df.info())

x = df[['O3', 'Solar', 'Wind']]
y = df['Temperature']
lm = LinearRegression()
lm.fit(x,y)

coefs = pd.DataFrame({'Feature' : ['O3', 'Solar', 'Wind'], 'coefs' : lm.coef_})

print(coefs)

answer_1 = round(coefs['coefs'][0],3)
print(answer_1)

x_test = pd.DataFrame({'O3' : [10], "Solar" : 90, "Wind" : 20})
pred = lm.predict(x_test)
print(pred)

answer_3 = round(pred[0], 3)
print(answer_3)

t_test = stats.ttest_ind(x['Wind'], y)
t_test.pvalue
answer_2 = round(t_test.pvalue,3)
print(answer_2)