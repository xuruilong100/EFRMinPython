import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import ConstantMean, GARCH
import seaborn as sb
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.api as tsa

# Exercise 1

spClose = pd.read_csv(
    'Chapter4_Data1.csv', parse_dates=True,
    index_col='Date', squeeze=True)
spClose.plot()

returns = spClose.apply(np.log) - spClose.shift(1).apply(np.log)
returns.dropna(inplace=True)
returns.plot()

# Method 1

garch = arch_model(returns)
rst = garch.fit(
    starting_values=np.array([0.0, 0.000005, 0.1, 0.85]))

type(rst)
rst.plot(annualize='D')

# Method 2

garch = ConstantMean(returns)
garch.volatility = GARCH(p=1, q=1)
rst = garch.fit(
    starting_values=np.array([0.0, 0.000005, 0.1, 0.85]))

rst.plot(annualize='D')

sb.distplot(rst.resid, fit=stats.norm)
