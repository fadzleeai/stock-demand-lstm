import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt

#Load train and test datasets
train_file_path = '/content/drive/MyDrive/DemandForecasting/train.csv'  #Replace with your train file path
test_file_path = '/content/drive/MyDrive/DemandForecasting/test.csv'    #Replace with your test file path

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

#Select features (including all relevant columns)
features = ['week', 'center_id', 'meal_id', 'checkout_price', 'base_price',
            'emailer_for_promotion', 'homepage_featured', 'num_orders']
train_data = train_df[features].copy()

if 'num_orders' in test_df.columns:
    test_data = test_df[features].copy()
else:
    test_df['num_orders'] = 0  
    test_data = test_df[features].copy()

# Apply log1p transformation to 'num_orders' before scaling
train_data['num_orders'] = np.log1p(train_data['num_orders'])
if 'num_orders' in test_data.columns:
    test_data['num_orders'] = np.log1p(test_data['num_orders'])

#Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

#Create sequences variables for LSTM
sequence_length = 10
X_train, y_train = [], []

#Create sequences for training
for i in range(sequence_length, len(train_scaled)):
    X_train.append(train_scaled[i-sequence_length:i, :-1])  #All but the last column (target)
    y_train.append(train_scaled[i, -1])  #Target column ('num_orders')

X_train, y_train = np.array(X_train), np.array(y_train)

# Create sequences for testing
X_test, y_test = [], []
for i in range(sequence_length, len(test_scaled)):
    X_test.append(test_scaled[i-sequence_length:i, :-1])
    y_test.append(test_scaled[i, -1])

X_test, y_test = np.array(X_test), np.array(y_test)

#Build the LSTM model
model = Sequential([
    LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),  #Add dropout after the first LSTM layer
    LSTM(50, activation='relu'),
    Dropout(0.2),  #Add dropout after the second LSTM layer
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

#Train the model
model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=1)

#Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")

#Predict and inverse transform
num_features = train_scaled.shape[1] - 1  # Total features excluding target

#Predicted orders
predictions = model.predict(X_test)
predicted_orders = scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], num_features)), predictions)))[:, -1]

#Actual orders
actual_orders = scaler.inverse_transform(
    np.hstack((np.zeros((y_test.shape[0], num_features)), y_test.reshape(-1, 1))))[:, -1]

#Convert predictions and actual values back to the original scale
predicted_orders = np.expm1(predicted_orders)
actual_orders = np.expm1(actual_orders)

#Plot results - Limiting the range to the first 100 data points
plt.figure(figsize=(10, 6))  # Set the figure size for better clarity
plt.plot(actual_orders[:100], label="Actual Orders")
plt.plot(predicted_orders[:100], label="Predicted Orders", linestyle='--')

#Add labels, legend, and title
plt.legend()
plt.title("LSTM Demand Forecasting (First 100 Data Points)")
plt.grid(True)  # Add gridlines for better readability

#Show the plot
plt.show()
