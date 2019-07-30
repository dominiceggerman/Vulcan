# By Dominic Eggerman

# Imports
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Convert array of values into a matrix
def createDataset(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        dataX.append(data[i:(i + look_back), 0])
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# If bitch != side
if __name__ == "__main__":
    # Import data
    df = pd.read_excel("EIA Weekly May 2019.xlsx", sheet_name="Query Results")
    df = df[["WeekEnding", "Lower48StocksBcf"]]
    df["WeekEnding"] = pd.to_datetime(df["WeekEnding"])
    df["Lower48StocksBcf"] = df["Lower48StocksBcf"].astype("float32")

    # Fix a seed so that results are reproducible.
    np.random.seed(8)
    
    # Normalize data
    data = df["Lower48StocksBcf"].values.reshape(len(df["Lower48StocksBcf"].values), 1)  # Reshape to fit in sklearn.preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Split into train and test datasets
    train = data[0:int(len(data)*0.67), :]
    test = data[int(len(data)*0.67):len(data), :]
    print(len(train), len(test))

    # Prepare train and test datasets for modelling
    # Arrays where y(t) = x(t+1)
    trainX, trainY = createDataset(train, 1)
    testX, testY = createDataset(test, 1)

    # Reshape input for memory block
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # Invert predictions (so performance is calculated in the same units as original data)
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    trainY = scaler.inverse_transform([trainY])
    testY = scaler.inverse_transform([testY])

    # Calculate RMSE
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print("Train Score (average error - Bcf): {0}".format(round(trainScore, 2)))
    print("Test Score (average error - Bcf): {0}".format(round(testScore, 2)))

    # Shift train predictions for plotting
    trainPredictPlot = np.empty_like(data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[1:len(trainPredict) + 1, :] = trainPredict

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (1*2) + 1:len(data)-1, :] = testPredict

    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(data))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()