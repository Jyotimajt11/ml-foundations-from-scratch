import pandas as pd;
import matplotlib.pyplot as plt;

from sklearn.datasets import fetch_california_housing;
from sklearn.linear_model import LinearRegression;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import mean_squared_error, r2_score;

#load the dataset
housing = fetch_california_housing();

# convert to dataframe
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['price'] = housing.target

#use all features instead of only median income, better model
X = data.drop('price', axis=1) # all features except price
y = data['price']

#split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LinearRegression()

#train the model
model.fit(X_train, y_train)

#predictions
y_pred = model.predict(X_test)

#evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Multiple Linear Regression Results")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("Intercept:", model.intercept_)

print("\nFeature Weights:")
for feature, weight in zip(X.columns, model.coef_):
    print(f"{feature}: {weight}")

# Actual vs Predicted plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Multiple Linear Regression: Actual vs Predicted")

plt.savefig("linear-regression-from-scratch/Outputs/multiple_regression_plot.png")
plt.show()



