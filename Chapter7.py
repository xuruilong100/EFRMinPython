import numpy as np
import pandas as pd
from scipy import stats, optimize
import seaborn as sns
import statsmodels.api as sm
from arch.univariate import ZeroMean, StudentsT
from archEx.NGARCH import NGARCH11

scale = 100

# Exercise 1

data = pd.read_csv(
    'data/Chapter7_Data.csv', parse_dates=True, index_col='date')

returns = data.apply(np.log) - data.apply(np.log).shift()
returns.dropna(inplace=True)
returns *= scale
returns.plot()

# Exercise 2

print(returns.cov())
print(returns.corr())

sns.regplot(
    x='sp500', y='tnote10',
    data=returns)

# Exercise 3

returns['portfolio'] = (returns['sp500'] + returns['tnote10']) / 2

std_sp = stats.norm.fit(returns['sp500'], floc=0)[1]
std_tn = stats.norm.fit(returns['tnote10'], floc=0)[1]
std_p = stats.norm.fit(returns['portfolio'], floc=0)[1]

VaRsp = -stats.norm.ppf(0.01, 0, scale=std_sp)
VaRtn = -stats.norm.ppf(q=0.01, scale=std_tn)
VaRp = -stats.norm.ppf(q=0.01, scale=std_p)

print(VaRp < (VaRsp + VaRtn) / 2)

# Exercise 4

omega = 1.5E-6 * scale ** 2
alpha = 0.05
beta = 0.8
theta = 1.25

tsm = ZeroMean(returns['sp500'])
ngarch = NGARCH11(
    np.array([omega, alpha, beta, theta]))
tsm.volatility = ngarch
tsm.distribution = StudentsT()
sp500_rst = tsm.fit()

print(sp500_rst)
sp500_rst.plot(annualize='D')

sns.distplot(sp500_rst.std_resid, fit=stats.t)

print(
    ngarch.is_valid(
        sp500_rst.params['alpha'],
        sp500_rst.params['beta'],
        sp500_rst.params['theta']))

sm.graphics.qqplot(
    sp500_rst.std_resid, line='45')

returns['std_sp500'] = sp500_rst.std_resid

omega = 5E-6 * scale ** 2
alpha = 0.03
beta = 0.97
theta = 0.0

tsm = ZeroMean(returns['tnote10'])
ngarch = NGARCH11(
    np.array([omega, alpha, beta, theta]))
tsm.volatility = ngarch
tsm.distribution = StudentsT()
tnote10_rst = tsm.fit()

print(tnote10_rst)
tnote10_rst.plot(annualize='D')

sns.distplot(tnote10_rst.std_resid, fit=stats.t)

print(
    ngarch.is_valid(
        tnote10_rst.params['alpha'],
        tnote10_rst.params['beta'],
        tnote10_rst.params['theta']))

sm.graphics.qqplot(
    tnote10_rst.std_resid, line='45')

returns['std_tnote10'] = tnote10_rst.std_resid

sns.regplot(
    x='std_sp500', y='std_tnote10',
    data=returns)

print(returns[['std_sp500', 'std_tnote10']].cov())
print(returns[['std_sp500', 'std_tnote10']].corr())


# Exercise 5

def ExponentialSmootherLogLikelihood(data: pd.DataFrame):
    z = np.asmatrix(data.values.T)

    def LogLikelihood(x: np.ndarray) -> float:
        # start value of Q
        Q = np.asmatrix(np.corrcoef(z))
        row_len, col_len = z.shape
        log_likelihood = 0.0
        lmd = x[0]

        for i in range(col_len):
            diag = np.asmatrix(np.diag(np.diag(Q)))
            sqrt_diag_inv = np.sqrt(diag.I)
            R = np.asmatrix(
                np.dot(np.dot(sqrt_diag_inv, Q), sqrt_diag_inv))

            core = np.log(np.linalg.det(R)) + np.dot(np.dot(z[:, i].T, R.I), z[:, i])

            # core is matrix
            log_likelihood += -0.5 * core[0, 0]

            # update Q
            Q = (1 - lmd) * np.dot(z[:, i], z[:, i].T) + lmd * Q

        return log_likelihood

    return LogLikelihood


logLikelihood = ExponentialSmootherLogLikelihood(
    returns[['std_sp500', 'std_tnote10']])

x = np.arange(0.9, 0.99, 0.005)
y = [logLikelihood(x[i:(i + 1)]) for i in range(len(x))]

logLikelihoodTab = pd.DataFrame(
    {'x': x, 'y': y})

sns.lineplot(
    x='x', y='y', data=logLikelihoodTab)

func = lambda x: -logLikelihood(x)
x0 = np.array([0.94])
res = optimize.minimize(
    func, x0, method='SLSQP', bounds=[(0.9, 0.99)])

print(res)


# Exercise 6


def MeanRevertingLogLikelihood(data: pd.DataFrame):
    z = np.asmatrix(data.values.T)

    def LogLikelihood(x: np.ndarray) -> float:
        # start value of Q and long-run correlation
        Q = np.asmatrix(np.corrcoef(z))
        ER = np.asmatrix(np.corrcoef(z))
        row_len, col_len = z.shape
        log_likelihood = 0.0
        alpha = x[0]
        beta = x[1]

        for i in range(col_len):
            diag = np.asmatrix(np.diag(np.diag(Q)))
            sqrt_diag_inv = np.sqrt(diag.I)
            R = np.asmatrix(
                np.dot(np.dot(sqrt_diag_inv, Q), sqrt_diag_inv))

            core = np.log(np.linalg.det(R)) + np.dot(np.dot(z[:, i].T, R.I), z[:, i])

            # core is matrix
            log_likelihood += -0.5 * core[0, 0]

            # update Q
            Q = (1 - alpha - beta) * ER + \
                alpha * np.dot(z[:, i], z[:, i].T) + \
                beta * Q

        return log_likelihood

    return LogLikelihood


logLikelihood = MeanRevertingLogLikelihood(
    returns[['std_sp500', 'std_tnote10']])

func = lambda x: -logLikelihood(x)
x0 = np.array([0.05, 0.9])
cons = optimize.LinearConstraint(
    A=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    lb=np.zeros(3),
    ub=np.ones(3))

res = optimize.minimize(
    func, x0, method='SLSQP',
    bounds=[(0.0, 0.2), (0.8, 1.0)],
    constraints=cons)

print(res)
