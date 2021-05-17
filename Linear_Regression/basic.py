import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(1)
x = np.random.rand(100,1)

# print(x)

y = 2 + 3*x + np.random.rand(100,1)
# print(y)

regression_model = LinearRegression()
regression_model.fit(x,y)
y_pre = regression_model.predict(x)

rmse = mean_squared_error(y, y_pre)
r2 = r2_score(y, y_pre)


plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, y_pre)
plt.show()

