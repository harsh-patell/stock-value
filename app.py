
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


st.title('Stock Trend Predictor')

#####################################
#        STOCK INFORMATION          #
#####################################
start = '2016-01-01'
end = datetime.today().strftime('%Y-%m-%d')
input_ticker = st.text_input('Enter Stock Ticker')

try:
    df = yf.download('AAPL', start, end)

    scaler = MinMaxScaler(feature_range=(0,1))
    closingData = df.filter(['Close'])
    dataset = closingData.values

    # Describe Raw Data
    st.subheader('Stock Data from 2016 - 2023')
    st.write(df.describe())

    #####################################
    #           VISUALIZATIONS          #
    ##################################### 
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize = (16,8))
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.plot(df['Close'], 'b')
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.plot(df.Close, 'b', label="Original Price")
    plt.plot(ma100, 'g', label="100 MA")
    plt.legend(loc='best')
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma200 = df.Close.rolling(200).mean()
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.plot(df['Close'], 'b')
    plt.plot(ma100, 'g')
    plt.plot(ma200, 'r', label="200 MA")
    plt.legend(loc='best')
    st.pyplot(fig)


    # Splitting Data into Training and Testing
    training_data_len = math.ceil(int(len(df) * 0.8))
    data_training = pd.DataFrame(df['Close'][0 : training_data_len]) 
    data_testing = pd.DataFrame(df['Close'][training_data_len: int(len(df))])

    print(data_training.shape) # Number of rows to train data on
    print(data_testing.shape)  # Number of rows to test data on

    # Loading the LSTM Model (training our data)
    model = load_model('keras_new_model.h5')


    #####################################
    #           TESTING DATA            #
    #####################################
    test_data = pd.concat([data_training.tail(100), data_testing])
    testing_data_array = scaler.fit_transform(test_data)

    x_test = []                          # feature data
    y_test = dataset[training_data_len:] # label data 

    for i in range(100, testing_data_array.shape[0]):
        x_test.append(testing_data_array[i-100: i])
        
    x_test = np.array(x_test)
    x_train = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    #####################################
    #           PREDICTIONS             #
    #####################################
    y_predicted = model.predict(x_test)
    y_predicted = scaler.inverse_transform(y_predicted)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(y_predicted - y_test)**2)
    print('RMSE: ', rmse)

    # Predictions Graph
    fig2 = plt.figure(figsize = (16,8))
    st.subheader('Complete Data Set')
    train = closingData[:training_data_len]
    test = closingData[training_data_len:]
    test['Predictions'] = y_predicted

    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.plot(train['Close'], 'b')
    plt.plot(test[['Close','Predictions']])
    plt.legend(['Train', 'Test', 'Predictions'], loc='best')
    st.pyplot(fig2)

    fig3 = plt.figure(figsize = (16,8))
    st.subheader('Predicted vs Original (Test Data)')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.plot(y_test, 'b', label="Original Price")
    plt.plot(y_predicted, 'c', label="Predicted Price")
    plt.legend(loc='best')
    st.pyplot(fig3)
    
except: 
    if (input_ticker != ""):
        st.write("Please enter a valid ticker.")
    
