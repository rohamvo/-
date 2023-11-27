import numpy as np

x = np.array([1,3,5,7,9])

def z_score(x) : 
    zscore = (x-np.mean(x)) / (np.std(x, ddof = 1))
    return zscore

x_z_score = z_score(x)

print(x_z_score)

###############################################################

import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/82108/Desktop/data/survey.csv')

test = pd.crosstab(df['Sex'], df['Exer'])

print(test)

from scipy.stats import chi2_contingency

result = chi2_contingency(test)

print(result)