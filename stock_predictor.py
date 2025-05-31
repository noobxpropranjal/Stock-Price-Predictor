import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime

# Step 1: Get historical stock data
ticker = input("Enter stock ticker symbol (e.g., AAPL for Apple): ")
start_date = "2020-01-01"
end_date = datetime.date.today().strftime('%Y-%m-%d')

data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    print("Error: No data found for this ticker.")
    exit()

# Step 2: Prepare data
data = data[['Close']]
data['Prediction'] = data['Close'].shift(-30)  # Predict 30 days into future

# Step 3: Features and labels
X = data[['Close']][:-30]
y = data['Prediction'][:-30]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict
future_days = 30
X_future = data[['Close']][-future_days:]
forecast = model.predict(X_future)

# Step 7: Show results
print("\nPredicted stock prices for the next 30 days:")
for i, price in enumerate(forecast):
    print(f"Day {i + 1}: ₹{round(price, 2)}")

# Step 8: Plot
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Historical Price')
plt.plot(range(len(data)-30, len(data)), forecast, label='Predicted Price', linestyle='dashed')
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
