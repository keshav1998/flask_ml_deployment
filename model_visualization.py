import pandas as pd
import numpy  as np
import pickle
import matplotlib.pyplot as plt
import load_data

plt.rcParams['figure.figsize'] = 25, 8
plt.rcParams['axes.grid'] = True
plt.style.use('fivethirtyeight')

X, y = load_data.create_feature_target()

model=pickle.load(open('model/prof.pkl','rb'))
x_test=np.array([[90000, 120000, 260000]])
print(x_test)
print(model.predict(x_test))

#To generate a best fit model
X_range=np.zeros((50,3))
y_range=np.zeros((50,))
for i in range(3):
    Xi=X.iloc[:,i].values
    vals=plt.hist(Xi,49)
    plt.xlabel("Feature")
    plt.ylabel("Frequency")
    X_range[:,i]=np.transpose(vals[1])
y_range=model.predict(X_range)    

# Plot the results
plt.figure()
plt.scatter(X.iloc[:,0].values, y, s=20, edgecolor="black", c="darkorange", label="train data")
plt.scatter(x_test[:,0], model.predict(x_test), s=30, color="yellowgreen", label="test data", linewidth=2)
plt.plot(X_range[:,0], y_range, color="cornflowerblue",
         label="Regression_model", linewidth=2)
plt.xlabel("R&D Cost")
plt.ylabel("Profit")
plt.title("Decision Tree Regression")
plt.legend()
plt.savefig("plots/RDvsProfit.png", dpi=300)
plt.show()

plt.figure()
plt.scatter(X.iloc[:,1].values, y, s=20, edgecolor="black", c="darkorange", label="train data")
plt.scatter(x_test[:,1], model.predict(x_test), s=30, color="yellowgreen", label="test data", linewidth=2)
plt.plot(X_range[:,1], y_range, color="cornflowerblue",
         label="Regression_model", linewidth=2)
plt.xlabel("Admin Cost")
plt.ylabel("Profit")
plt.title("Decision Tree Regression")
plt.legend()
plt.savefig("plots/ADvsProfit.png", dpi=300)
plt.show()

plt.figure()
plt.scatter(X.iloc[:,2].values, y, s=20, edgecolor="black", c="darkorange", label="train data")
plt.scatter(x_test[:,2], model.predict(x_test), s=30, color="yellowgreen", label="test data", linewidth=2)
plt.plot(X_range[:,2], y_range, color="cornflowerblue",
         label="Regression_model", linewidth=2)
plt.xlabel("Marketing Cost")
plt.ylabel("Profit")
plt.title("Decision Tree Regression")
plt.legend()
plt.savefig("plots/MCvsProfit.png", dpi=300)
plt.show()