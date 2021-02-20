import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import load_data

plt.rcParams['figure.figsize'] = 25, 8
plt.rcParams['axes.grid'] = True
plt.style.use('fivethirtyeight')

X, y = load_data.create_feature_target()
reg=DecisionTreeRegressor(max_depth=5)
reg.fit(X,y)
y_pred=reg.predict(X)
print(reg.score(X, y))