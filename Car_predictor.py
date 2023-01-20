import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

# Data collection and processing
car_data_set = pd.read_csv("/Users/ben/Documents/DATA/car data.csv")
car_data_set.head()


# Collecting indormation on the data
car_data_set.info()

# Encoding the categorical data

# Encoding Fuel_Typle column
car_data_set.replace(
    {'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)

# Encoding Seller_Typle column
car_data_set.replace(
    {'Seller_Type': {'Dealer': 0, 'Individual': 1, }}, inplace=True)

# Encoding Transmission column
car_data_set.replace(
    {'Transmission': {'Manual': 0, 'Automatic': 1, }}, inplace=True)


# Splitting the data into traiining data and testing data

x = car_data_set.drop(["Car_Name", "Selling_Price"], axis=1)
y = car_data_set["Selling_Price"]

print(x)
print(y)

# Splitting training and test data
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.1, random_state=2)

# Modeling Training

# Loading the linear regression model
lin_reg_model = LinearRegression()

lin_reg_model.fit(X_train, Y_train)

# Prediction on Training  data
training_data_prediction = lin_reg_model.predict(X_train)

# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

# Visualize the actual prices and Predicted
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")
plt.title("Actual Price vs Predicted Prices")
plt.show()

# Prediction on Training  data
test_data_prediction = lin_reg_model.predict(X_test)

# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)

# Visualize the actual prices and Predicted
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")
plt.title("Actual Price vs Predicted Prices")
plt.show()


# Lasso Regression
lass_reg_model = Lasso()
lin_reg_model.fit(X_train, Y_train)

# Prediction on Training  data
training_data_prediction = lin_reg_model.predict(X_train)

# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

# Visualize the actual prices and Predicted
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")
plt.title("Actual Price vs Predicted Prices")
plt.show()

# Prediction on Training  data
test_data_prediction = lin_reg_model.predict(X_test)
