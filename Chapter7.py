import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl

mpl.use('TkAgg')

import seaborn as sns

scale = 100

# Exercise 1

data = pd.read_csv(
    'data/Chapter7_Data.csv', parse_dates=True, index_col='date')

returns = data.apply(np.log) - data.apply(np.log).shift()
returns.dropna(inplace=True)
returns *= scale
returns.plot()

# Exercise 2

print(returns.cov())
print(returns.corr())

sns.regplot(
    x='sp500', y='tnote10',
    data=returns)

# Exercise 3

returns['portfolio'] = (returns['sp500'] + returns['tnote10']) / 2

std_sp = stats.norm.fit(returns['sp500'], floc=0)[1]
std_tn = stats.norm.fit(returns['tnote10'], floc=0)[1]
std_p = stats.norm.fit(returns['portfolio'], floc=0)[1]

VaRsp = -stats.norm.ppf(0.01, 0, scale=std_sp)
VaRtn = -stats.norm.ppf(q=0.01, scale=std_tn)
VaRp = -stats.norm.ppf(q=0.01, scale=std_p)

print(VaRp < (VaRsp + VaRtn) / 2)
