import numpy as np; # perform mathematical operations like mean, median, standard deviation, etc.
import pandas as pd; # read and manipulate data, handle missing values, perform data cleaning, etc.
import matplotlib.pyplot as plt; # visualize data, create scatter plots, line plots, histograms, etc.

# Load the dataset
data = pd.read_csv("linear-regression-from-scratch/data/housing_sample.csv"); # read the dataset from the csv File

# Display the first few rows of the dataset
print(data.head()); # print the first 5 rows of the dataset to understand its structure and
# Check for missing values
print(data.isnull().sum()); # check for missing values in each column of the dataset

X = data["area"].values #area column as input feature (independent variable)
y = data["price"].values #price column as target variable (dependent variable)

X_mean = np.mean(X); # calculate the mean of the input feature
X_std = np.std(X); # calculate the standard deviation of the input feature

y_mean = np.mean(y); # calculate the mean of the target variable
y_std = np.std(y); # calculate the standard deviation of the target variable

X_scaled = (X - X_mean) / X_std; # standardize the input feature by subtracting the mean and dividing by the standard deviation
y_scaled = (y - y_mean) / y_std; # standardize the target variable by subtracting the mean and dividing by the standard deviation

weight = 0;
bias = 0;

learning_rate = 0.01; # set the learning rate for gradient descent
epochs = 1000; # set the number of iterations for training the model

n = len(X_scaled); # get the number of samples in the dataset
loss_history = []; # initialize an empty list to store the loss values during training

for epoch in range(epochs): # loop through the specified number of epochs
    y_pred = weight * X_scaled + bias; # calculate the predicted value using the current weights and bias
    loss = (1/n) * np.sum((y_scaled - y_pred) ** 2); # calculate the mean squared error loss
    loss_history.append(loss); # append the current loss value to the loss history list

    dw = -(2/n) * np.sum(X_scaled * (y_scaled - y_pred)); # calculate the gradient with respect to weights
    db = -(2/n) * np.sum(y_scaled - y_pred); # calculate the gradient with respect to bias
  
    weight  = weight - learning_rate * dw; # update the weights using the calculated gradient and learning rate
    bias = bias - learning_rate * db; # update the bias using the calculated gradient and learning_rate
     
    if epoch % 100 == 0: # print the loss value every 100 epochs to monitor the training process
        print(f"Epoch {epoch}, Loss: {loss:.4f}"); # print the current epoch number and the corresponding loss value

scaled_predictions = weight * X_scaled + bias; # calculate the predicted values using the final weights and bias after training
predictions = scaled_predictions * y_std + y_mean; # convert the scaled predictions back to the original scale by multiplying with the standard deviation and adding the mean

print("\n final weight :" , weight); # print the final weight after training
print("final bias :" , bias); # print the final bias after training

plt.scatter(X,y, color='green', label='Actual Data'); # create a scatter plot of the original data points
plt.plot(X, predictions, color='red', label='Predicted Line'); # plot the predicted line based on the final weights and bias
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression From Scratch")
plt.legend()
plt.savefig("linear-regression-from-scratch/Outputs/regression_line.png")
plt.show()


plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Reduction During Training")
plt.savefig("linear-regression-from-scratch/Outputs/loss_curve.png")
plt.show()




