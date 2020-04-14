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
    'Chapter2_Data1.csv', index_col='Date',
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
