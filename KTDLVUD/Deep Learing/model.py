import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to get valid integer input for number of days to predict
def get_valid_input(message):
    while True:
        try:
            user_input = input(message)
            if user_input == '':
                raise ValueError("Input cannot be empty.")
            return int(user_input)
        except ValueError:
            print("Please enter a valid number.")

# Function to get a valid date input for prediction
def get_valid_date_input():
    while True:
        try:
            target_date_str = input("Enter the target date to predict until (dd-mm-yyyy): ")
            if target_date_str == '':
                raise ValueError("Date cannot be empty.")
            target_date = pd.to_datetime(target_date_str, format="%d-%m-%Y")
            return target_date
        except ValueError:
            print("Invalid date format. Please enter a valid date (dd-mm-yyyy).")

# Load dataset
file_path = 'usd_vnd_exchange_rate_adjusted_data_filtered.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Print the column names to identify the correct name for the exchange rate
print(data.columns)

# Select only the 'Exchange Rate' column using the correct column name
exchange_rate_data = data['USD_to_VND'].values.reshape(-1, 1)

# Scale the data for better LSTM performance
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(exchange_rate_data)

# Define train and test size
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 30  # Using 30 days of data to predict the next day
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Reverse the scaling
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual Exchange Rate', color='blue')
plt.plot(predictions, label='Predicted Exchange Rate', color='red')
plt.title('USD/VND Exchange Rate Prediction')
plt.xlabel('Time (days)')
plt.ylabel('Exchange Rate (VND/USD)')
plt.legend()
plt.show()

# Function to predict future exchange rates
def predict_future(days):
    last_sequence = scaled_data[-sequence_length:]
    predicted_rates = []
    predicted_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)
    for _ in range(days):
        prediction = model.predict(last_sequence.reshape(1, sequence_length, 1))
        predicted_rates.append(prediction[0][0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)

    predicted_rates = scaler.inverse_transform(np.array(predicted_rates).reshape(-1, 1))
    return predicted_dates, predicted_rates

# User input for prediction
future_days = get_valid_input("Enter the number of days to predict into the future: ")
predicted_dates, future_predictions = predict_future(future_days)

# Print the predicted values
print(f"Predicted Exchange Rates for next {future_days} days:")
for date, rate in zip(predicted_dates, future_predictions):
    print(f"{date.strftime('%d-%m-%Y')}: {rate[0]:.2f} VND/USD")

# Function to predict future exchange rates until a specific date
def predict_until_date(target_date):
    last_sequence = scaled_data[-sequence_length:]
    predicted_rates = []
    current_date = data.index[-1] + pd.Timedelta(days=1)
    while current_date <= target_date:
        prediction = model.predict(last_sequence.reshape(1, sequence_length, 1))
        predicted_rates.append((current_date, prediction[0][0]))
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
        current_date += pd.Timedelta(days=1)
    predicted_rates = [(date, scaler.inverse_transform([[rate]])[0][0]) for date, rate in predicted_rates]
    return predicted_rates

# User input for prediction
target_date = get_valid_date_input()

# Predict until the target date
predicted_values = predict_until_date(target_date)

# Print the predicted value for the target date
if predicted_values:
    date, rate = predicted_values[-1]
    print(f"Predicted Exchange Rate for {target_date.strftime('%d-%m-%Y')}: {rate:.2f} VND/USD")
else:
    print(f"No prediction available for {target_date.strftime('%d-%m-%Y')}")

# Calculate and print the evaluation metrics
test_mse = mean_squared_error(y_test_actual, predictions)
test_mae = mean_absolute_error(y_test_actual, predictions)

print(f"Mean Squared Error (MSE) on test data: {test_mse:.2f}")
print(f"Mean Absolute Error (MAE) on test data: {test_mae:.2f}")
