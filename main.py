import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Dataset Load பண்ணுதல்
df = pd.read_csv('data.csv')
data = df['Temp'].values.reshape(-1, 1)

# 2. Data Normalization (0 - 1 Scale)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Time Series Sequences உருவாக்குதல் (Sliding Window)
def create_sequences(dataset, time_step=3):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 3 # கடந்த 3 நாட்களை வைச்சு அடுத்த நாளை predict பண்ணும்
X, y = create_sequences(scaled_data, time_step)

# Reshape for LSTM: [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# 4. LSTM Model Build பண்ணுதல்
model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_step, 1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Model Train பண்ணுதல்
print("Training Model...")
model.fit(X, y, epochs=50, batch_size=2, verbose=1)

# 6. Prediction & Inverse Scale (Original Value-க்கு மாற்றுதல்)
predictions = model.predict(X)
predicted_values = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y.reshape(-1, 1))

# 7. Result Plotting
plt.figure(figsize=(8,4))
plt.plot(actual_values, label='Actual Temp', color='blue', marker='o')
plt.plot(predicted_values, label='Predicted Temp', color='red', linestyle='--', marker='x')
plt.title('LSTM Temperature Prediction')
plt.xlabel('Samples')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.show()