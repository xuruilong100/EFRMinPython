import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from arch.univariate import ZeroMean, RiskMetrics2006, StudentsT, GARCH
from copulas.multivariate import GaussianMultivariate

scale = 100

# Exercise 1

data = pd.read_csv(
    'data/Chapter9_Data.csv', parse_dates=True, index_col='date')

returns = data.apply(np.log) - data.apply(np.log).shift()
returns.dropna(inplace=True)
returns *= scale
returns.plot()

percentile = range(15, 86)

cor0 = pd.DataFrame(
    index=percentile, columns=['cor'])

for p in percentile:

    score_sp = stats.scoreatpercentile(returns['sp'], p)
    score_tn = stats.scoreatpercentile(returns['tn'], p)

    if p <= 50:
        cut = returns.loc[
            (returns['sp'] <= score_sp) & (returns['tn'] <= score_tn),]

        cor_num = stats.pearsonr(cut['sp'], cut['tn'])

        cor0.loc[p, 'cor'] = cor_num[0]
    else:
        cut = returns.loc[
            (returns['sp'] > score_sp) & (returns['tn'] > score_tn),]

        cor_num = stats.pearsonr(cut['sp'], cut['tn'])

        cor0.loc[p, 'cor'] = cor_num[0]

cor0.plot()

tsm_sp = ZeroMean(returns['sp'])
garch = GARCH()
tsm_sp.volatility = garch
tsm_sp.distribution = StudentsT()
rst_sp = tsm_sp.fit()

filtered_sp = rst_sp.std_resid

tsm_tn = ZeroMean(returns['tn'])
garch = GARCH()
tsm_tn.volatility = garch
tsm_tn.distribution = StudentsT()
rst_tn = tsm_tn.fit()

filtered_tn = rst_tn.std_resid

filtered_returns = pd.DataFrame(
    dict(sp=filtered_sp, tn=filtered_tn),
    index=returns.index)

cor1 = pd.DataFrame(
    index=percentile, columns=['cor'])

for p in percentile:

    score_sp = stats.scoreatpercentile(filtered_sp, p)
    score_tn = stats.scoreatpercentile(filtered_tn, p)

    if p <= 50:
        cut = filtered_returns.loc[
            (filtered_returns['sp'] <= score_sp) & (filtered_returns['tn'] <= score_tn),]

        cor_num = stats.pearsonr(cut['sp'], cut['tn'])

        cor1.loc[p, 'cor'] = cor_num[0]
    else:
        cut = filtered_returns.loc[
            (filtered_returns['sp'] > score_sp) & (filtered_returns['tn'] > score_tn),]

        cor_num = stats.pearsonr(cut['sp'], cut['tn'])

        cor1.loc[p, 'cor'] = cor_num[0]

cor1.plot()

# Exercise 2

n = 100000
rho = [-0.3, 0.0, 0.3, 0.6, 0.9]
percentile = range(10, 91)

cor = pd.DataFrame(
    index=percentile,
    columns=['rho=' + str(r) for r in rho])

for r in rho:
    mnorm = stats.multivariate_normal(
        mean=np.array([0.0, 0.0]),
        cov=np.array([[1.0, r], [r, 1.0]]))

    rv = pd.DataFrame(
        mnorm.rvs(size=n),
        index=range(n), columns=['n1', 'n2'])

    for p in percentile:

        score1 = stats.scoreatpercentile(rv['n1'], p)
        score2 = stats.scoreatpercentile(rv['n2'], p)

        idx = 'rho=' + str(r)

        if p <= 50:
            cut = rv.loc[
                (rv['n1'] <= score1) & (rv['n2'] <= score2),]

            cor_num = stats.pearsonr(cut['n1'], cut['n2'])

            cor.loc[p, idx] = cor_num[0]
        else:
            cut = rv.loc[
                (rv['n1'] > score1) & (rv['n2'] > score2),]

            cor_num = stats.pearsonr(cut['n1'], cut['n2'])

            cor.loc[p, idx] = cor_num[0]

cor.plot()

# Exercise 3 & 4

data = pd.read_csv(
    'data/Chapter9_Data.csv', parse_dates=True, index_col='date')

returns = data.apply(np.log) - data.apply(np.log).shift()
returns.dropna(inplace=True)
returns *= scale

tsm_sp = ZeroMean(returns['sp'])
rm2006 = RiskMetrics2006()
tsm_sp.volatility = rm2006
tsm_sp.distribution = StudentsT()
rst_sp = tsm_sp.fit()

tsm_tn = ZeroMean(returns['tn'])
tsm_tn.volatility = rm2006
tsm_tn.distribution = StudentsT()
rst_tn = tsm_tn.fit()

filtered_returns = pd.DataFrame(
    dict(sp=rst_sp.std_resid, tn=rst_tn.std_resid),
    index=returns.index)

f = sns.PairGrid(returns)
f.map_diag(sns.distplot)
f.map_offdiag(sns.scatterplot)

g = sns.PairGrid(filtered_returns)
g.map_diag(sns.distplot)
g.map_offdiag(sns.scatterplot)

var_sp = rst_sp.forecast().variance.dropna().values[0]
var_tn = rst_tn.forecast().variance.dropna().values[0]

vol_sp = np.sqrt(var_sp)
vol_tn = np.sqrt(var_tn)

vol = np.array([vol_sp, vol_tn])

n = 10000

## copulas package

gaus_cop1 = GaussianMultivariate()
gaus_cop1.fit(filtered_returns)

print(gaus_cop1.covariance)

samples1 = gaus_cop1.sample(n).values
scale_samples1 = samples1 * vol.T

sim_returns1 = 0.5 * (scale_samples1[:, 0] + scale_samples1[:, 1])

sorted_returns1 = np.sort(sim_returns1)

sns.distplot(sorted_returns1)

var = stats.scoreatpercentile(sorted_returns1, 1)
es = np.mean(sorted_returns1[:100])
