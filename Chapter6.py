import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels.api as sm
from scipy import stats
from arch.univariate import ConstantMean
from archEx.NGARCH import NGARCH11, FixedNGARCH11

# Exercise 1

spClose = pd.read_csv(
    'data/Chapter1_Data.csv', index_col='Date',
    squeeze=True, parse_dates=True)

spClose.plot()

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns.plot()

scaledReturns = returns / returns.std()
scaledReturns.plot()

sm.graphics.qqplot(
    scaledReturns, line='45')

# Exercise 2-3

scale = 100.0

spClose = pd.read_csv(
    'data/Chapter4_Data1.csv', parse_dates=True,
    index_col='Date', squeeze=True)
spClose.plot()

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns *= scale
returns.dropna(inplace=True)
returns.plot()

omega = 0.000005 * scale ** 2
alpha = 0.07
beta = 0.85
theta = 0.5

# using NGARCH11

tsm = ConstantMean(returns)
ngarch = NGARCH11(
    np.array([omega, alpha, beta, theta]))
tsm.volatility = ngarch
rst = tsm.fit()

print(rst)
rst.plot(annualize='D')

sb.distplot(rst.resid, fit=stats.norm)

print(
    ngarch.is_valid(
        rst.params['alpha'],
        rst.params['beta'],
        rst.params['theta']))

sm.graphics.qqplot(
    rst.std_resid, line='45')

# using FixedNGARCH11

tsm = ConstantMean(returns)
fixed_ngarch = FixedNGARCH11(
    1.373877,  # author's result
    np.array([omega, alpha, beta]))
tsm.volatility = fixed_ngarch
rst = tsm.fit()

print(rst)
rst.plot(annualize='D')

sb.distplot(rst.resid, fit=stats.norm)

sm.graphics.qqplot(
    rst.std_resid, line='45')
