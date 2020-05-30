import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels.api as sm

# Exercise 1

data = pd.read_csv(
    'data/Chapter5_Data1.csv',
    parse_dates=True,
    index_col='Date')

print(data.head())

close = data['Close']
returns = close.apply(np.log) - close.apply(np.log).shift()
returns2 = returns ** 2
sigma2 = data['sigma2']

lm = sm.OLS(
    returns2,
    exog=sm.add_constant(sigma2),
    missing='drop').fit()

print(lm.summary())

sb.regplot(x=sigma2, y=returns2)

# Exercise 2

data = pd.read_csv(
    'data/Chapter5_Data2.csv',
    parse_dates=True,
    index_col='Date')

print(data.head())

d = data['High'].apply(np.log) - data['Low'].apply(np.log)
rp = d ** 2 / (4.0 * np.log(2.0))
sigma2 = data['sigma2']

lm = sm.OLS(
    rp,
    exog=sm.add_constant(sigma2),
    missing='drop').fit()

print(lm.summary())

sb.regplot(x=sigma2, y=rp)

# Exercise 3

data = pd.read_csv(
    'data/Chapter5_Data3.csv',
    parse_dates=True,
    index_col='Date')

print(data.head())

rv = data['RVaverage']
sigma2 = data['sigma2']

lm = sm.OLS(
    rv,
    exog=sm.add_constant(sigma2),
    missing='drop').fit()

print(lm.summary())

sb.regplot(x=sigma2, y=rv)

# Exercise 4

data = pd.read_csv(
    'data/Chapter5_Data2.csv',
    parse_dates=True,
    index_col='Date')

print(data.head())

d = data['High'].apply(np.log) - data['Low'].apply(np.log)
rpDaily = d ** 2 / (4.0 * np.log(2.0))
rpWeekly = rpDaily.rolling(5).mean()
rpMonthly = rpDaily.rolling(21).mean()

y = rpDaily.apply(np.log)
x = pd.DataFrame({
    'cons': np.ones(y.shape),
    'lnRPdaily': rpDaily.apply(np.log).shift(),
    'lnRPweekly': rpWeekly.apply(np.log).shift(),
    'lnRPmonthly': rpMonthly.apply(np.log).shift()})

lm = sm.OLS(
    y,
    exog=x,
    missing='drop').fit()

print(lm.summary())

# Exercise 5

data = pd.read_csv(
    'data/Chapter5_Data3.csv',
    parse_dates=True,
    index_col='Date')

print(data.head())

rvDaily = data['RVaverage']
rvWeekly = rvDaily.rolling(5).mean()
rvMonthly = rvDaily.rolling(21).mean()

y = rvDaily.apply(np.log)
x = pd.DataFrame({
    'cons': np.ones(y.shape),
    'lnRVdaily': rvDaily.apply(np.log).shift(),
    'lnRVweekly': rvWeekly.apply(np.log).shift(),
    'lnRVmonthly': rvMonthly.apply(np.log).shift()})

lm = sm.OLS(
    y,
    exog=x,
    missing='drop').fit()

print(lm.summary())
