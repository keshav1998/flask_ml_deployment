import pandas as pd
import numpy  as np
import pickle
from sklearn.tree import DecisionTreeRegressor
import load_data

X, y = load_data.create_feature_target()
reg=DecisionTreeRegressor(max_depth=5)
reg.fit(X,y)
y_pred=reg.predict(X)
print(reg.score(X, y))

pickle.dump(reg, open('model/prof.pkl', 'wb'))