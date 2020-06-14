import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from arch.univariate import ConstantMean, StudentsT, ZeroMean, GARCH
from prettytable import PrettyTable
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

sns.distplot(rst.std_resid, fit=stats.norm)

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

sns.distplot(rst.std_resid, fit=stats.norm)

sm.graphics.qqplot(
    rst.std_resid, line='45')

# Exercise 4

scale = 100.0

spClose = pd.read_csv(
    'data/Chapter4_Data1.csv', parse_dates=True,
    index_col='Date', squeeze=True)

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns *= scale
returns.dropna(inplace=True)

omega = 0.000005 * scale ** 2
alpha = 0.07
beta = 0.85
theta = 0.5

# using NGARCH11

tsm = ZeroMean(returns)
ngarch = NGARCH11(
    np.array([omega, alpha, beta, theta]))
tsm.volatility = ngarch
tsm.distribution = StudentsT()
rst = tsm.fit(
    starting_values=np.array(
        [omega, alpha, beta, theta, 10.0]))

print(rst)
rst.plot(annualize='D')

sns.distplot(rst.std_resid, fit=stats.t)

print(
    ngarch.is_valid(
        rst.params['alpha'],
        rst.params['beta'],
        rst.params['theta']))

sm.graphics.qqplot(
    rst.std_resid, line='45')

# using FixedNGARCH11

tsm = ZeroMean(returns)
fixed_ngarch = FixedNGARCH11(
    1.373877,  # author's result
    np.array([omega, alpha, beta]))
tsm.volatility = fixed_ngarch
tsm.distribution = StudentsT()
rst = tsm.fit(
    starting_values=np.array(
        [omega, alpha, beta, 10.0]))

print(rst)
rst.plot(annualize='D')

sns.distplot(rst.std_resid, fit=stats.t)

sm.graphics.qqplot(
    rst.std_resid, line='45')

# Exercise 5

scale = 100.0

spClose = pd.read_csv(
    'data/Chapter4_Data1.csv', parse_dates=True,
    index_col='Date', squeeze=True)

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns *= scale
returns.dropna(inplace=True)


def HillEstimator(resid, num):
    sortedResid = np.sort(resid)
    tailLoss = -sortedResid[0:num]
    u = -sortedResid[num + 1]

    xi = np.mean(np.log(tailLoss)) - np.log(u)
    c = num / len(resid) * u ** (1.0 / xi)
    est = {'xi': xi, 'c': c, 'u': u}

    return est


def HillPpf(q, hill_est):
    return 1.0 / (q / hill_est['c']) ** hill_est['xi']


def CornishFisherPpf(q, resid):
    nppf = stats.norm.ppf(q)
    s = stats.skew(resid)
    k = stats.kurtosis(resid)
    cfp = -(nppf + 0.74 * s - 0.24 * k + 0.38 * s ** 2)

    return cfp


# using GARCH(1,1)-t as filter

tsm = ConstantMean(returns)
garch = GARCH(p=1, q=1)
tsm.volatility = garch
tsm.distribution = StudentsT()
rst = tsm.fit()

print(rst)

sns.distplot(rst.std_resid, fit=stats.t)

sm.graphics.qqplot(
    rst.std_resid, line='45')

hillEst = HillEstimator(rst.std_resid, 50)
print(hillEst)

tailProb = 1 / 100.0

nu = rst.params['nu']

tab = PrettyTable(['Model', 'Tail Quantile'])
tab.add_row(['normal', -stats.norm.ppf(tailProb)])
tab.add_row(['standardized t', -StudentsT().ppf(tailProb, nu)])
# tab.add_row([
#     'standardized t',
#     stats.t.ppf(
#         tailProb, df=nu, scale=1 / stats.t.std(df=nu))])
tab.add_row(['Hill', HillPpf(tailProb, hillEst)])
tab.add_row(['Cornish-Fisher', CornishFisherPpf(tailProb, rst.std_resid)])

print(tab)

# using NGARCH(1,1)-t as filter

omega = 0.000005 * scale ** 2
alpha = 0.07
beta = 0.85
theta = 0.5

tsm = ZeroMean(returns)
ngarch = NGARCH11(
    np.array([omega, alpha, beta, theta]))
tsm.volatility = ngarch
tsm.distribution = StudentsT()
rst = tsm.fit()

print(rst)

tailProb = 1 / 100.0

nu = rst.params['nu']

hillEst = HillEstimator(rst.std_resid, 50)
print(hillEst)

tab = PrettyTable(['Model', 'Tail Quantile'])
tab.add_row(['normal', -stats.norm.ppf(tailProb)])
tab.add_row(['standardized t', -StudentsT().ppf(tailProb, nu)])
# tab.add_row([
#     'standardized t',
#     stats.t.ppf(
#         tailProb, df=nu, scale=1 / stats.t.std(df=nu))])
tab.add_row(['Hill', HillPpf(tailProb, hillEst)])
tab.add_row(['Cornish-Fisher', CornishFisherPpf(tailProb, rst.std_resid)])

print(tab)

# using FixedNGARCH(1,1)-t filter

tsm = ZeroMean(returns)
fixed_ngarch = FixedNGARCH11(
    1.373877,  # author's result
    np.array([omega, alpha, beta]))
tsm.volatility = fixed_ngarch
tsm.distribution = StudentsT()
rst = tsm.fit(
    starting_values=np.array(
        [omega, alpha, beta, 10.0]))

print(rst)

tailProb = 1 / 100.0

nu = rst.params['nu']

hillEst = HillEstimator(rst.std_resid, 50)
print(hillEst)

tab = PrettyTable(['Model', 'Tail Quantile'])
tab.add_row(['normal', -stats.norm.ppf(tailProb)])
tab.add_row(['standardized t', -StudentsT().ppf(tailProb, nu)])
# tab.add_row([
#     'standardized t',
#     stats.t.ppf(
#         tailProb, df=nu, scale=1 / stats.t.std(df=nu))])
tab.add_row(['Hill', HillPpf(tailProb, hillEst)])
tab.add_row(['Cornish-Fisher', CornishFisherPpf(tailProb, rst.std_resid)])

print(tab)

# Exercise 6

scale = 100.0

spClose = pd.read_csv(
    'data/Chapter4_Data1.csv', parse_dates=True,
    index_col='Date', squeeze=True)

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns *= scale
returns.dropna(inplace=True)

# using GARCH(1,1)-t as filter

tsm = ConstantMean(returns)
garch = GARCH(p=1, q=1)
tsm.volatility = garch
tsm.distribution = StudentsT()
rst = tsm.fit()

print(rst)

sm.graphics.qqplot(
    rst.std_resid, line='45')

hillEst = HillEstimator(rst.std_resid, 50)

empTailProb = (np.array(range(50)) + 0.5) / len(returns)
empTailQtl = np.sort(rst.std_resid)[0:50]
evtTailQtl = -HillPpf(empTailProb, hillEst)

f, ax = plt.subplots()
ax.set_xlim(-8.0, 0.0)
ax.set_ylim(-8.0, 0.0)

sns.scatterplot(
    x=evtTailQtl, y=empTailQtl, ax=ax)
sns.lineplot(
    x=[-8, 0], y=[-8, 0], ax=ax)

# using NGARCH(1,1)-t as filter

omega = 0.000005 * scale ** 2
alpha = 0.07
beta = 0.85
theta = 0.5

tsm = ZeroMean(returns)
ngarch = NGARCH11(
    np.array([omega, alpha, beta, theta]))
tsm.volatility = ngarch
tsm.distribution = StudentsT()
rst = tsm.fit()

print(rst)

hillEst = HillEstimator(rst.std_resid, 50)

empTailProb = (np.array(range(50)) + 0.5) / len(returns)
empTailQtl = np.sort(rst.std_resid)[0:50]
evtTailQtl = -HillPpf(empTailProb, hillEst)

f, ax = plt.subplots()
ax.set_xlim(-8.0, 0.0)
ax.set_ylim(-8.0, 0.0)

sns.scatterplot(
    x=evtTailQtl, y=empTailQtl, ax=ax)
sns.lineplot(
    x=[-8, 0], y=[-8, 0], ax=ax)

# using FixedNGARCH(1,1)-t as filter

omega = 0.000005 * scale ** 2
alpha = 0.07
beta = 0.85

tsm = ZeroMean(returns)
fixed_ngarch = FixedNGARCH11(
    1.373877,  # author's result
    np.array([omega, alpha, beta]))
tsm.volatility = fixed_ngarch
tsm.distribution = StudentsT()
rst = tsm.fit()

print(rst)

hillEst = HillEstimator(rst.std_resid, 50)

empTailProb = (np.array(range(50)) + 0.5) / len(returns)
empTailQtl = np.sort(rst.std_resid)[0:50]
evtTailQtl = -HillPpf(empTailProb, hillEst)

f, ax = plt.subplots()
ax.set_xlim(-8.0, 0.0)
ax.set_ylim(-8.0, 0.0)

sns.scatterplot(
    x=evtTailQtl, y=empTailQtl, ax=ax)
sns.lineplot(
    x=[-8, 0], y=[-8, 0], ax=ax)
