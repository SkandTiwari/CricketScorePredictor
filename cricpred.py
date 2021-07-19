import numpy as nump
import pandas as pnd
import matplotlib as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression

data = pnd.read_csv('Sampled score.csv')
print(data)

#Datavisualisation part
fig = px.scatter(data, x = 'Overs', y='Scores')
#fig.show()
formula = LinearRegression()
x = data.Overs.values.reshape(-1, 1)
y = data.Scores.values.reshape(-1, 1)
#print(x)

#applying formula for linear regression
formula.fit(x, y)
SixOver = formula.predict([[6]])
print(SixOver)