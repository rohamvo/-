import pandas as pd
import numpy as np

df = pd.read_csv("data/Titanic.csv")

from scipy.stats import chi2_contingency
from scipy.stats import levene
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp, wilcoxon

test = pd.crosstab(df['Gender'], df['Survived'])

result = chi2_contingency(test)

answer_1 = round(result[0], 3)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty = 'none')
df['Gender'] = df['Gender'].map({'female' : 1, 'male' : 0})
x = pd.get_dummies(df[['Gender', 'SibSp', 'Parch', 'Fare']])
df['Survived'] = df['Survived'].astype('category')
y = df['Survived']

from sklearn.model_selection import train_test_split

model.fit(x, y)

print(model.coef_)

answer_2 = round(model.coef_[0][2], 3)

answer_3 = round(np.exp(model.coef_[0][2]), 3)

print(answer_1, answer_2, answer_3)
print(model.intercept_)

#######################################################################################

import pandas as pd

df1 = pd.read_csv("data/customer_train.csv")
df2 = pd.read_csv("data/customer_test.csv")

df1['성별'] = df1['성별'].astype('category')
y = df1['성별']
df1['환불금액'] = df1['환불금액'].fillna(0)
df1 = df1.drop('회원ID', axis = 1)
df1 = df1.drop('성별', axis = 1)
x = pd.get_dummies(df1)

df2['환불금액'] = df2['환불금액'].fillna(0)
df2 = df2.drop('회원ID', axis = 1)
x_test = pd.get_dummies(df2)

common_feature = list(set(x.columns).intersection(x_test.columns))

x = x[common_feature]
x_test = x_test[common_feature]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3, stratify = y)

model = RandomForestClassifier(n_estimators = 300)

model.fit(x_train, y_train)

pred = model.predict(x_valid)

print(roc_auc_score(y_valid, pred))
print(accuracy_score(y_valid, pred))
print(precision_score(y_valid, pred))
print(recall_score(y_valid, pred))

pred_test = model.predict(x_test)

print(len(pred_test))