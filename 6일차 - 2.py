import pandas as pd
import numpy as np
from scipy.stats import chisquare

# 위약 샘플 데이터가 부작용 없음인 데이터를 0~1 사이의 확률로 출력하시오. 반올림하여 소수점 셋째자리로 출력
# 답 : 0.787

# 카이제곱 검정으로 검정 통계량을 출력하시오. 반올림하여 소수점 셋쨰 자리로 출력
# 답 : 0.997

# 유의확률(p-value)를 출력하시오.(반올림하여 소수점 셋째 자리로 출력)
# 답 : 0.802

df = pd.read_csv('C:/Users/82108/Desktop/data/P230605.csv', encoding='euc-kr')

n_df = df.groupby('코드').size().reset_index(name='n')

n_df['n'][3]

answer_1 = n_df['n'][3] / np.sum(n_df['n'])

round(answer_1, 3)

r_df = pd.DataFrame({'코드' : [1,2,3,4], '비율' : [0.05, 0.1, 0.05, 0.8]})
total = np.sum(n_df['n'])

result = chisquare(n_df['n'], f_exp = r_df['비율'] * total)

answer_2 = round(result.statistic, 3)
answer_3 = round(result.pvalue, 3)

print(answer_2)
print(answer_3)