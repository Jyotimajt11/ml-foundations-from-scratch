import pandas as pd;
import matplotlib.pyplot as plt;

from sklearn.datasets import fetch_california_housing;
from sklearn.linear_model import LinearRegression;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import mean_squared_error, r2_score;

#load the dataset
housing = fetch_california_housing();
# convert to dataframe

data = pd.DataFrame(housing.data, columns=housing.feature_names);
data['price'] = housing.target;

print(data.head());

#select features and the target variable
X = data[['MedInc']] # median income
y = data['price'] # house price

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42);

#create model
model = LinearRegression();

#train the model
model.fit(X_train, y_train);

#predictions
y_pred = model.predict(X_test);

# Evaluate the model
mse = mean_squared_error(y_test, y_pred);
r2 = r2_score(y_test, y_pred);
print("\nModel Results:");
print("weight:", model.coef_[0]);
print("bias:", model.intercept_);
print("Mean Squared Error:", mse);
print("R2 Score:", r2);

#plotting the results
plt.scatter(X_test, y_test, color="green", label ="Actual_data");
plt.scatter(X_test, y_pred, color="red", label = "Predicted data");
plt.xlabel("Median Income");
plt.ylabel("House Price");

plt.title("Linear Regression on California Housing Dataset");
plt.legend();

# save the output plot
plt.savefig("linear-regression-from-scratch/Outputs/real_dataset_plot.png")

plt.show();


# The model is showing underfitting: Price depends on more features like age, income, location, etc. and not just median income. 
# To improve the model, we can include more features from the dataset and also consider using more complex models like polynomial regression or decision trees.




