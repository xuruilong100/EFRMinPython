import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from prettytable import PrettyTable
from scipy import stats

spClose = pd.read_csv(
    'Chapter1_Data.csv', index_col='Date',
    squeeze=True, parse_dates=True)

print(type(spClose))

spClose.plot()

# Exercise 1

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns.plot()

# Exercise 2

print(returns.mean())
print(returns.std())
print(returns.skew())
print(returns.kurt())

sb.distplot(returns, fit=stats.norm)

# Exercise 3

print(tsa.acf(
    returns.dropna(), nlags=100))

sm.graphics.tsa.plot_acf(
    returns.dropna(), lags=100)

# Exercise 4

returns2 = returns ** 2
returns2.plot()

print(tsa.acf(
    returns2.dropna(), nlags=100))

sm.graphics.tsa.plot_acf(
    returns2.dropna(), lags=100)

# Exercise 5

sigma0 = returns.std()

sigma2 = pd.Series(
    data=sigma0 ** 2, index=returns2.index)

# returns2[0] is NaN
for i in range(2, len(sigma2)):
    sigma2[i] = 0.94 * sigma2[i - 1] + 0.06 * returns2[i - 1]

sigma2.plot()

# Exercise 6

z = returns / sigma2.apply(np.sqrt)
z.plot()

print(z.mean())
print(z.std())
print(z.skew())
print(z.kurt())

# Exercise 7

spClose5Day = spClose[[i % 5 == 0 for i in range(len(spClose))]]
spClose10Day = spClose[[i % 10 == 0 for i in range(len(spClose))]]
spClose15Day = spClose[[i % 15 == 0 for i in range(len(spClose))]]

returnsDaily = returns
returns5Day = spClose5Day.apply(np.log) - spClose5Day.shift(1).apply(np.log)
returns10Day = spClose10Day.apply(np.log) - spClose10Day.shift(1).apply(np.log)
returns15Day = spClose15Day.apply(np.log) - spClose15Day.shift(1).apply(np.log)

sb.distplot(returnsDaily, fit=stats.norm)
sb.distplot(returns5Day, fit=stats.norm)
sb.distplot(returns10Day, fit=stats.norm)
sb.distplot(returns15Day, fit=stats.norm)

tab = PrettyTable(['item', 'Daily', '5-Day', '10-Day', '15-Day'])
tab.add_row(['mean', returnsDaily.mean(), returns5Day.mean(), returns10Day.mean(), returns15Day.mean()])
tab.add_row(['std', returnsDaily.std(), returns5Day.std(), returns10Day.std(), returns15Day.std()])
tab.add_row(['skew', returnsDaily.skew(), returns5Day.skew(), returns10Day.skew(), returns15Day.skew()])
tab.add_row(['kurt', returnsDaily.kurt(), returns5Day.kurt(), returns10Day.kurt(), returns15Day.kurt()])

tab.float_format = '.4'
print(tab)

# Exercise 7

q = -stats.norm.ppf(0.01)
VaR = q * sigma2.apply(np.sqrt)
returns.plot()
VaR.plot()

returnsVaR = pd.DataFrame(
    dict(returns=returns, VaR=VaR))

returnsVaR.plot()
