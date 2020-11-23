import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from arch.univariate import ZeroMean, EWMAVariance, StudentsT, GARCH, Normal
from archEx.NGARCH import NGARCH11

scale = 100

# Exercise 1 & 3

data = pd.read_csv(
    'data/Chapter8_Data.csv', parse_dates=True, index_col='date')

returns = data.apply(np.log) - data.apply(np.log).shift()
returns.dropna(inplace=True)
returns *= scale
returns.plot()

# FHS

fhs = ZeroMean(returns['Close'])
garch = GARCH(p=1, q=1)
fhs.distribution = StudentsT()
fhs.volatility = garch
rst = fhs.fit()

print(rst)

rst.plot(annualize='D')

sns.distplot(rst.std_resid, fit=stats.t)

forecast_variance_1day = rst.forecast(
    horizon=1).variance.iloc[-1, 0]

rs = np.random.RandomState(1234)


def Bootstrap(samples,
              random_state: np.random.RandomState,
              size: int):
    max_idx = len(samples)
    i = 0
    while i < size:
        yield samples[random_state.randint(0, max_idx)]
        i += 1


sim1 = np.array([x for x in Bootstrap(rst.std_resid, rs, 10000)])
sim1.sort()

var1 = np.sqrt(forecast_variance_1day * 10.0) * abs(sim1[100])
es1 = np.sqrt(forecast_variance_1day * 10.0) * abs(sim1[0:100].mean())

sim2 = np.array(
    [np.sum([x for x in Bootstrap(rst.std_resid, rs, 10)]) for i in range(10000)])
sim2.sort()

var2 = np.sqrt(forecast_variance_1day) * abs(sim2[100])
es2 = np.sqrt(forecast_variance_1day) * abs(sim2[0:100].mean())

print('forecast variance(1 day):', forecast_variance_1day)
print('forecast VaR(method 1):', var1)
print('forecast ES(method 1):', es1)
print('forecast VaR(method 2):', var2)
print('forecast ES(method 2):', es2)

'''
forecast variance(1 day): 0.3631740572871991
forecast VaR(method 1): 4.8197260341789825
forecast ES(method 1): 6.370822956022516
forecast VaR(method 2): 4.3080501562825395
forecast ES(method 2): 5.092286290645029
'''

print(sm.graphics.tsa.acf(rst.std_resid, nlags=5))
print(sm.graphics.tsa.pacf(rst.std_resid, nlags=5))

# RiskMetrics

riskMetrics = ZeroMean(returns['Close'])
ewma = EWMAVariance(lam=0.94)
riskMetrics.volatility = ewma
rst = riskMetrics.fit()

print(rst)

rst.plot(annualize='D')

forecast_variance_10days = rst.forecast(
    horizon=1).variance.iloc[-1, 0] * 10.0

print('forecast variance(10 days):', forecast_variance_10days)
print('forecast VaR:', -np.sqrt(forecast_variance_10days) * stats.norm.ppf(0.01))
print('forecast ES:', np.sqrt(forecast_variance_10days) * stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01)

'''
forecast variance(10 days): 3.6425481901571626
forecast VaR: 4.439942146600405
forecast ES: 5.086684187983292
'''

# NGARCH(1,1)

omega = 1.5E-6 * scale ** 2
alpha = 0.05
beta = 0.8
theta = 1.25

ngarch = ZeroMean(returns['Close'])
ngarch11 = NGARCH11(
    np.array([omega, alpha, beta, theta]))
ngarch.volatility = ngarch11
ngarch.distribution = Normal()
rst = ngarch.fit()

print(rst)

rst.plot(annualize='D')

print(
    ngarch11.is_valid(
        rst.params['alpha'],
        rst.params['beta'],
        rst.params['theta']))

rs = np.random.RandomState(1234)
forecast_variance_10days = rst.forecast(
    horizon=10,
    method='simulation',
    rng=lambda x: rs.normal(size=x),
    simulations=1000).variance.iloc[-1, :]

forecast_variance_10days = forecast_variance_10days.sum()

print('forecast variance(10 days):', forecast_variance_10days)
print('forecast VaR:', -np.sqrt(forecast_variance_10days) * stats.norm.ppf(0.01))
print('forecast ES:', np.sqrt(forecast_variance_10days) * stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01)

'''
forecast variance(10 days): 3.618636152135496
forecast VaR: 4.425344827214484
forecast ES: 5.069960557076489
'''
