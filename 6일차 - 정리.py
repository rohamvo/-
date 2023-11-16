# 코드 정리

# TXT 파일 데이터 수집

f = open('data.txt', 'r') # 파일을 읽기 모드('r')로 열고 f 에 저장
while True :
    line = f.readline() # f에서 한줄을 읽어 line에 저장
    if not line : # 더 이상 줄이 없을때 break
        break
    print(line) # line 출력
f.close() # 파일 닫기

# csv 파일 데이터 수집
import pandas as pd
a = pd.read_csv('data.csv')

# csv로 저장
a.to_csv('data.csv', index = False) # 파일명 지정 및 인덱스 설정

# 엑셀 파일 데이터 수집
import pandas as pd

a = pd.read_excel('data.xlsx')

# 엑셀로 저장
a.to_excel(excel_writer='data.xlsx', sheet_name = 'data_sheet', index = False) # 파일명, 시트명, 인덱스 설정

# 데이터 전처리

# 데이터프레임 선택
df = pd.DataFrame({'ex' : [1,2,3,4,5], 'exx' : [6,7,8,9,10]})
print(df)
df.iloc[:,1]
df.iloc[1,1] # [행, 열]
df[['ex']] # 컬럼 선택
df[df['ex']==1] # 조건에 맞는 행 출력
df.loc[df['ex']==1]

# 파생 변수

def add(a, b) : 
    while True :
        return a + b
    
a = df[['ex', 'exx']].apply(add)

# 원 핫 인코딩
pd.get_dummies(x, drop_first = True) # drop_first = 첫 번째 더미 변수 제거 여부 지정

# groupby
test = pd.DataFrame({'test' : [1,2,3,1,2,1,2], 'a' : [1,2,5,6,7,8,9]})
test.groupby('test').sum()
test.groupby('test').mean()
test.groupby('test').max()
test.groupby('test').var() # 분산
test.groupby('test').std() # 표준편차
test.groupby('test').median() # 중앙값
test.groupby('test').size() # 빈도

pd.concat(a,b, axis = 0) # 데이터 병합 / axis = 0 행일 기준으로 결합 즉 위아래로 붙이기 / 1은 열을 기준으로 옆으로 붙이기

pd.merge(x,y, how = 'inner', on='abc') # abc열을 기준으로 공통으로 존재하는 모든 열 병합 sql생각하면댐
# how에는 inner(교집합), outer(합집합), left(left join), right(right join)

np.where(조건) # numpy배열에서 값 찾기

# 결측값 확인
df.isna()
df.info()

# 결측값 삭제
df.drop('a', axis = 1) # axis  0 은 행 기준 삭제, 1은 열 기준 삭제

df.dropna(axis = , how = all or any) # axis기준에 따라 삭제하며 how에 any이면 axis 기준에 결측값 하나라도 있으면 제거
# all이면 axis 기준에 따라 모든 값이 결측값인 경우에만 제거

# 단순대치법
df.fillna('대채할값', axis = ?)

# 이상값
# ESD = 평균으로부터 3 표준편차 떨어진 값을 이상값으로 판단 /평균-3표준편차<DATA<평균+3표준편차/ 이상값 민감
# 기하평균 활용 = 기하평균으로부터 2.5 표준편차 떨어진 값을 이상값으로 판단
# 사분위수 활용 = 사분위수 범위(Q3-Q1) 활용 Q1-1.5사분위수범위 < DATA < Q3 + 1.5사분위수범위 제외 이상값

# 사분위수
import matplotlib.pyplot as plt
x = pd.DataFrame({'abc' : [100,200,300,400,400,500,100,200,250,300,350,900,3]})
a = plt.boxplot(x) # 박스플롯 그리기
minimum = a['whiskers'][0].get_ydata()[1]
q1 = a['boxes'][0].get_ydata()[1]
q2 = a['medians'][0].get_ydata()[0]
q3 = box_score['whiskers'][1].get_ydata()[1]
maximum = box_score['whiskers'][1].get_ydata()[1]

# IQR(사분위수 범위계산 함수)
from scipy.stats import iqr
iqr(x)

astype.('category') # 범주형으로 변환

datetime.strptime(date_string, format)

# 데이터 정규화
# 최소 - 최대 정규화
from sklearn.preprocessing import MinMaxScaler
scale = MinmaxScaler()


random.choice(a, size, replace, prob) # 배열, 추출 개수, 복원 추출 여부, 

np.mean(array) # 배열 평균
np.nanmean(array) # 결측값 자동 제거 후 평균
np.trim_mean(array, proportiontocut) # 양끝 proportiontocut 퍼센테이지 만큼 제거 후 평균

np.bincount(array).argmax() # bincount로 각 요소 빈도 계산하고 argmax()로 최대값 출력

np.var(array, ddof) # 분산 / ddof는 자유도
np.std(array, ddof) # 표준편차
np.ptp(array) # 범위 계산 / 최대값 - 최소값

np.percentile(array, q) # 백분위 계산 / q는 계산할 백분위수 값
# ex) np.percentile(x, 25) 는 x배열의 25% 값이니 q1임

# 순위 계산
pd.Series(x).rank(method) 
# method에는 책 참고 4-56

from sklearn.metrics import mean_squared_error # 평균제곱오차(MSE)
from sklearn.metrics import r2_score # 결정계수 r제곱
model.predict(x) # 예측 값 반환
model.predict_proba(x) # 각 범주에 속할 확률을 반환하는 함수
from sklearn.metrics import counfusion_matrix # 혼동행렬
confusion_matrix(y_valid, pred)
from sklearn.metrics import accuracy_score # 정확도
from sklearn.metrics import f1_score # f1 스코어
from sklearn.metrics import roc_auc_score # roc_auc_score
from sklearn.metrics import recall_socre # 재현율 (실제 positive 중 맞춘 비율)
from sklearn.mercics import precision_score # 정밀도(positive라고 예측한 것중 맞은비율)

from sklearn.linear_model import LinearRegression # 선형회귀
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀분석
from sklearn.tree import DecisionTreeClassifier # 의사결정나무
from sklearn.svm import SVC # 서포트 벡터 머신(한번 다시 보자 4-98)
from sklearn.neighbors import KneighborClassifier # KNN 알고리즘
from sklearn.neural_network import MLPClassifier # 인공신경망
from sklearn.ensemble import BaggingClassifier # 배깅
from sklearn.ensemble import RandomForestClassifier # 랜덤포레스트 분류 / 종속변수가 범주형
from sklearn.ensemble import RandomForestRegression # 랜덤포레스트 회귀 / 종속변수가 수치형

from scipy.stats import shapiiro, wilcoxon, ttest_1samp
shapiro(x) # 정규성 검정 샤피로 검정 결과로 나오는 statistic은 검정통계량, pvalue는 p-값
ttest_1samp(x, popmean) # 정규성 가정을 만족하면 시행하는 t검정 / x는 표본으로부터 관측한 값 / popmean은 검정 기준 값
wilcoxon(x, alternative) # 정규성 가정을 만족하지 않을 때 wilcoxn수행 / x 는 표본으로부터 관측한 값 /
# alternative는 tow-sided(양측검정), less(좌측검정), greater(우측검정)으로 나뉜다 자세한건 책 4-134

from scipy.stats import levene
ttest_rel # 쌍체검정 / 처치 받기 전과 후의 차이를 알아보기위해 사용 / 쌍체표본 t검정은 표본이 하나 독립변수가 하나일때 사용

from scipy.stats import levene
levene(sample1, sample2, center) # 등분산성 검정 함수 / center는 분포의 중심을 정의하는 방법을 지정 median과 mean이 들어감
# 독립표본 t-검정
# 데이터가 서로 다른 모집단에서 추출된 경우 사용
# 독립된 두 집단의 평균차이를 검정
# 정규성, 등분산성 가정이 만족되는지 확인해야함

# 10개 미만의 표본은 정규성을 가정하지 못한다고 가정하고 Mann_Whitney Test 적용
# 10~30개의 표본은 샤피로 월크 검정, 콜모고로프-스미르노프 검정등의 방법을 통해서 정규성 증명
# 30개 이상은 중심극한정리에 의해 자동 만족
from scipy.stats import ttest_ind
ttest_ind(sample1, sample2, alternative, equal_var)
# alternative에는 tow-sided(양측검정), less(좌측검정), greater(우측검정)이 있고 equal_var은 등분산성 만족 여부이다.


f.cdf(x, dfn, dfd) # f검정은 두 표본의 분산에 대한 차이가 통계적으로 유의한가를 판별하는 검정기법 
# dfn = f분포의 분자의 자유도, dfd = f분포의 분모의 자유도 / 공식 외워야 함

# 적합도 검정 - 표본 집단의 분포가 주어진 특정 분포를 따르고 있는지를 검정하는 기법
# 적합도 검정 자유도 공식은 자유도 = 범주의 수 -1 이다
from scipy.stats import chisquare
chisquare(f_obs, f_exp) # 관찰 빈도를 나타내는 데이터, 기대 빈도를 나타내는 데이터

# 독립성 검정 - 각 범주가 서로 독립적인지, 서로 연관성이 있는지를 검정하는 기법
# 독립성 검정 자유도 = 자유도 = (범주1의수-1) X (범주2의수 -1)
from scipy.stats import chi2_contingency
chi2_contingency(observed) # 두 개 이상의 변수를 포함하는 2차원 배열
# 동질성 검정은 얼마나 서로 비슷한지 
# 검정 함수는 독립성 검정과 같음
# 동질성 검정에서의 귀무가설은 모집단은 동질하다로 설정


