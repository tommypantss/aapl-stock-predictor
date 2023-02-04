import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data from Yahoo Finance
data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1577836800&period2=1609459200&interval=1d&events=history')

# Plot the closing price over time to visualize the data
plt.plot(data['Close'])
plt.xlabel('Year')
plt.ylabel('Closing Price (USD)')
plt.title('AAPL Stock Price Over Time')
plt.show()

# Prepare the data for training
X = np.array(data.index).reshape(-1, 1)
y = np.array(data['Close']).reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the stock price on the test data
y_pred = regressor.predict(X_test)

# Plot the predicted vs actual stock price
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.xlabel('Year')
plt.ylabel('Closing Price (USD)')
plt.title('AAPL Stock Price Prediction')
plt.show()
