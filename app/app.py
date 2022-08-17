
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# Get the stock information (scraping from Yahoo Finance)
start = '2016-01-01'
end = '2022-08-08'

st.title('Stock Trend Predictor')
input_ticker = st.text_input('Enter Stock Ticker', 'TSLA')


df = data.DataReader(input_ticker, 'yahoo', start, end)
scaler = MinMaxScaler(feature_range=(0,1))


# Describe Raw Data
st.subheader('Stock Data from 2016 - 2022')
st.write(df.describe())


# Visualizations 
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(df['Close'], 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(ma100, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.plot(df['Close'], 'b')
st.pyplot(fig)


# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) 
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape) # Number of rows to train data on
print(data_testing.shape)  # Number of rows to test data on


# Loading the LSTM Model (training our data)
model = load_model('keras_model.h5')


# Scaling the testing data after appending it to last 100 days of training data
past_100_days = data_training.tail(100)
test_data = past_100_days.append(data_testing, ignore_index=True)
testing_data_array = scaler.fit_transform(test_data)

x_test = [] 
y_test= []  

for i in range(100, testing_data_array.shape[0]):
    x_test.append(testing_data_array[i-100: i])
    y_test.append(testing_data_array[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)
x_train = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# print(x_test.shape)
# print(x_test)
# print(y_test.shape)


# Predictions
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Predictions Graph
st.subheader('Predicted vs Original Data')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'c', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)