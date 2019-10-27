from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics

HeadBrainData = pd.read_csv('headbrain.csv')
print("\nBrief Information About Data :")
print(HeadBrainData.info())

# Don't need Gender,Age Columns
print("\nDropping Unnecessary Columns(Gender, Age )....")
HeadBrainData.drop(['Gender', 'Age Range'], inplace=True, axis=1)

print("Now We Have : \n")
print(HeadBrainData.head())

# Getting only first column
X = HeadBrainData.iloc[:, :-1].values
# Getting only second column
Y = HeadBrainData.iloc[:, 1].values

# Splitting Data int Train And Test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Applying Linear Model
Model = LinearRegression()
Model.fit(x_train, y_train)

# Value of M and C
print("\nWe got Coefficient value of our Best Fit Regression Line : ", Model.coef_)
print("and Intercept as : ", Model.intercept_)

# Prediction......
y_pred = Model.predict(x_test)

# Plotting.....
plt.title("Head Size Vs Brain Weight")
plt.xlabel("Head Size")
plt.ylabel("Brain Weight")
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, 'r')
plt.show()

# Comparision between Actual and Predicted Values
print("Comparision between Actual and Predicted Values :")
ModelRepresentation = pd.DataFrame(data={'Actual Y': y_test, 'Predicted Y': y_pred})
print(ModelRepresentation.head(10))

# Evaluating Some Errors
"""
Mean Absolute Error (MAE) is the mean of the absolute value of the errors. It is calculated as:

        MAE= E(|Actual - predict|)/n


Mean Squared Error (MSE) is the mean of the squared errors and is calculated as:

         MSE= E(|Actual - predict|**2)/n

Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

        MSE= sqrt(E(|Actual - predict|**2)/n)

Root Mean Squared Error
"""

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Score of Our Model
print("Model Score : ", Model.score(x_train, y_train))