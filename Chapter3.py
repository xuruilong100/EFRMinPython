import pandas as pd
import seaborn as sb
import statsmodels.api as sm
import statsmodels.tsa.api as tsa

# Exercise 4

dataSet = pd.read_csv('Chapter3_Data1.csv')

dataSet.plot()

arParams = pd.Series(
    data=0.0, index=dataSet.columns)

for c in dataSet.columns:
    model = tsa.AutoReg(dataSet[c], lags=1).fit()
    arParams.loc[c] = model.params[1]

sb.distplot(arParams)

# Exercise 5

dataSet = pd.read_csv('Chapter3_Data2.csv', squeeze=True)

dataSet.plot()

sm.graphics.tsa.plot_acf(dataSet)
sm.graphics.tsa.plot_pacf(dataSet)

model = tsa.ARMA(dataSet, order=(0, 1)).fit()

print(model.params)
print(model.sigma2)
