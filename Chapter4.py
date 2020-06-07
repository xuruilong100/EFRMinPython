import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, ZeroMean, StudentsT
import seaborn as sb
from scipy import stats
import statsmodels.api as sm
from archEx.NGARCH import NGARCH11, FixedNGARCH11
from archEx.GARCHX import GARCHX

scale = 100.0

# Exercise 1

spClose = pd.read_csv(
    'data/Chapter4_Data1.csv', parse_dates=True,
    index_col='Date', squeeze=True)
spClose.plot()

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns *= scale
returns.dropna(inplace=True)
returns.plot()

# Method 1

omega = 0.000005 * scale ** 2
alpha = 0.1
beta = 0.85

garch = arch_model(returns)
rst = garch.fit(
    starting_values=np.array([0.0, omega, alpha, beta]))

print(rst)
rst.plot(annualize='D')

# Method 2

tsm = ConstantMean(returns)
garch = GARCH(p=1, q=1)
tsm.volatility = garch
rst = tsm.fit(
    starting_values=np.array([0.0, omega, alpha, beta]))

print(rst)
rst.plot(annualize='D')

sb.distplot(rst.resid, fit=stats.norm)

# Exercise 2

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

returns2 = returns ** 2.0
filtered_returns2 = rst.std_resid ** 2.0

sm.graphics.tsa.plot_acf(returns2, lags=100)
sm.graphics.tsa.plot_acf(filtered_returns2, lags=100)

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

returns2 = returns ** 2.0
filtered_returns2 = rst.std_resid ** 2.0

sm.graphics.tsa.plot_acf(returns2, lags=100)
sm.graphics.tsa.plot_acf(filtered_returns2, lags=100)

# Exercise 3

spClose = pd.read_csv(
    'data/Chapter4_Data1.csv', parse_dates=True,
    index_col='Date', squeeze=True)

vix = pd.read_csv(
    'data/Chapter4_Data2.csv', parse_dates=True,
    index_col='Date', squeeze=True)

vix *= scale
vix2 = vix ** 2.0 / 252.0

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns *= scale
returns.dropna(inplace=True)

# Correlation of sigma and vix

tsm = ConstantMean(returns)
garch = GARCH(p=1, q=1)
tsm.volatility = garch
tsm.distribution = StudentsT()
rst = tsm.fit()

sigma2 = rst.conditional_volatility ** 2.0

print(rst)

sb.regplot(
    x=vix2,
    y=sigma2,
    marker='.')

print(stats.pearsonr(vix2, sigma2))

'''
The correlation of sigma2 and vix2 is really high.

Because of multicollinearity, vix2 is not a good explanatory 
variable for (N)GARCHX model.

Maybe, we can replace sigma2 by vix2 in (N)GARCHX model.
'''

# using GARCHX(1,1)-normal

tsm = ZeroMean(returns)
garchx = GARCHX(vix2.values)
tsm.volatility = garchx
rst = tsm.fit()

print(rst)
rst.plot(annualize='D')

# using GARCHX(1,0)-normal

tsm = ZeroMean(returns)
garchx = GARCHX(vix2.values, q=0)
tsm.volatility = garchx
rst = tsm.fit()

print(rst)
rst.plot(annualize='D')
