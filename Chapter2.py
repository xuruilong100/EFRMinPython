import numpy as np
import pandas as pd
from scipy import stats


def HSmethod(data, p):
    return -stats.scoreatpercentile(
        data, p * 100)


def WHSmethod(data, p, eta):
    m = len(data)
    eta_m = eta ** m
    weight = pd.Series(
        data=[(eta ** (m - 1 - i)) * (1 - eta) / (1 - eta_m) for i in range(m)],
        index=data.index)
    weightData = pd.DataFrame(
        dict(data=data, weight=weight))
    weightData.sort_values(
        by='data', inplace=True)

    weightCumsum = weightData['weight'].cumsum()
    per = stats.percentileofscore(
        weightCumsum, p)

    return -stats.scoreatpercentile(
        weightData['data'], per)


spClose = pd.read_csv(
    'data/Chapter2_Data1.csv', index_col='Date',
    squeeze=True, parse_dates=True)

print(type(spClose))

spClose.plot()

# Exercise 1

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns = returns.dropna()
returns.plot()

r87oct = returns['1987-10']

longHsVaR = pd.DataFrame(
    dict(
        loss=-r87oct,
        hsVaR=[HSmethod(returns[:r87oct.index[i]][-251:-1], 0.01) for i in range(len(r87oct))]),
    index=r87oct.index)

longHsVaR.plot()

print(longHsVaR)

shortHsVaR = pd.DataFrame(
    dict(
        loss=r87oct,
        hsVaR=[HSmethod(-returns[:r87oct.index[i]][-251:-1], 0.01) for i in range(len(r87oct))]),
    index=r87oct.index)

shortHsVaR.plot()

print(shortHsVaR)

# Exercise 2

longWhsVaR = pd.DataFrame(
    dict(
        loss=-r87oct,
        hsVaR=[WHSmethod(returns[:r87oct.index[i]][-251:-1], 0.01, 0.99) for i in range(len(r87oct))]),
    index=r87oct.index)

longWhsVaR.plot()

print(longWhsVaR)

shortWhsVaR = pd.DataFrame(
    dict(
        loss=r87oct,
        hsVaR=[WHSmethod(-returns[:r87oct.index[i]][-251:-1], 0.01, 0.99) for i in range(len(r87oct))]),
    index=r87oct.index)

shortWhsVaR.plot()

print(shortWhsVaR)

# Exercise 3

spClose = pd.read_csv(
    'data/Chapter2_Data2.csv', index_col='Date',
    squeeze=True, parse_dates=True)

print(type(spClose))

spClose.plot()

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns = returns.dropna()
returns.plot()

returns2 = returns ** 2
returns2.plot()

lmbd = 0.94
var = pd.Series(
    data=0.0, index=returns2.index)

for i in range(1, len(var)):
    var[i] = lmbd * var[i - 1] + (1 - lmbd) * returns2[i - 1]

rmVaR = -np.sqrt(10) * stats.norm.ppf(0.01) * var.apply(np.sqrt)
rmVaR.plot()

returnsCrisis = returns['2008-07-01':'2009-12-31']
returnsCrisis.plot()

hsVaR = pd.Series(
    data=[np.sqrt(10) * HSmethod(returns[:returnsCrisis.index[i]][-251:-1], 0.01) for i in range(len(returnsCrisis))],
    index=returnsCrisis.index)

whsVaR = pd.Series(
    data=[np.sqrt(10) * WHSmethod(returns[:returnsCrisis.index[i]][-251:-1], 0.01, 0.99) for i in range(len(returnsCrisis))],
    index=returnsCrisis.index)

returnsVaR = pd.DataFrame(
    dict(
        returns=returns['2008-07-01':'2009-12-31'],
        rmVaR=rmVaR['2008-07-01':'2009-12-31'],
        hsVaR=hsVaR,
        whsVaR=whsVaR))

returnsVaR.plot()

# Exercise 4

N = 100000
simpleReturns = spClose / spClose.shift(1) - 1

positionsRm = N / rmVaR
pnlDailyRm = positionsRm * simpleReturns['2008-07-01':'2009-12-31']
pnlRm = pnlDailyRm.dropna().cumsum()

pnlRm.plot()

positionsHs = N / hsVaR
pnlDailyHs = positionsHs * simpleReturns['2008-07-01':'2009-12-31']
pnlHs = pnlDailyHs.dropna().cumsum()

pnlHs.plot()

positionsWhs = N / whsVaR
pnlDailyWhs = positionsWhs * simpleReturns['2008-07-01':'2009-12-31']
pnlWhs = pnlDailyWhs.dropna().cumsum()

pnlWhs.plot()

pnls = pd.DataFrame(
    dict(
        pnlRm=pnlRm,
        pnlHs=pnlHs,
        pnlWhs=pnlWhs))

pnls.plot()
