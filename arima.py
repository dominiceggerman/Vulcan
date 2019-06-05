# By Dominic Eggerman

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from pyramid.arima import auto_arima, ARIMA
import sklearn.metrics as skmetrics


# If bitch != side
if __name__ == "__main__":
    # Import data (excel)
    df = pd.read_excel("EIA Weekly May 2019.xlsx", sheet_name="Query Results")
    df = df[["WeekEnding", "Lower48StocksBcf"]]
    df["WeekEnding"] = pd.to_datetime(df["WeekEnding"])

    # Plot storage inventory
    plt.plot(df.set_index("WeekEnding"))
    plt.xlabel("Date")
    plt.ylabel("Lower 48 Inventory (Bcf)")
    plt.show()

    # Decompose the time series using seasonal_decompose ?? Residuals ??
    decomp = seasonal_decompose(df.set_index("WeekEnding"), model="multiplicative")
    decomp.plot()
    plt.show()
    # Residuals seem a touch high, but multiplicative model seems to fit well ??


    # Introduce AutoRegressive Integrated Moving Average (ARIMA)
    ## AutoRegressive : model using dependent relationship between an observation and some number of lagged observations
    ## Integrated : use of differencing of raw observations in order to make time series stationary
    ## Moving Average : model using the dependency between an observation and residual error from a moving average model

    # ARIMA notation is ARIMA(p,d,q) where:
    ## p : the number of lag observations (past values) included in the model
    ## d : the number of times the raw observation are differenced (degrees of differencing)
    ## q : size of the moving average window

    # We can add a seasonal element to the ARIMA model (SARIMA), with elements SARIMA(P,D,Q), which will predict the seasonal component


    # Use Augmented Dickey Fuller test to check if series is stationary
    adf_result = adfuller(df["Lower48StocksBcf"])
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    # p-value = X, so the series has to be differenced

    # Plot Auto-correlations with differencing. How does this work ??
    fig, axes = plt.subplots(3, 1, sharex=True)
    plot_acf(df.set_index("WeekEnding"), ax=axes[0])
    plot_acf(df.set_index("WeekEnding").diff().dropna(), ax=axes[1])
    plot_acf(df.set_index("WeekEnding").diff().diff().dropna(), ax=axes[2])
    # pd.plotting.autocorrelation_plot(df.set_index("WeekEnding"))
    plt.show()
    # First-order differencing the positive correlation is 'significant' for the first 10 lags (-), hits blue at 8


    # Akaike information criterion (AIC) ??
    ## An estimator of the relative quality of statistical models for a given set of data
    ## Compares how well a model fits the data
    ### Models with better fits and fewer features have lower scores

    # # Generate multiple ARIMA fits (m is the period for seasonal differencing - 52 is weekly)
    # # This shit takes forever (1h+) to run with m=52
    # stepwise = auto_arima(df.set_index("WeekEnding"),
    #                         start_p=1, start_q=1, max_p=12, max_q=18, d=0,
    #                         start_P=0, D=1, start_Q=1,
    #                         seasonal=True, m=12, trace=True,
    #                         error_action="ignore", suppress_warnings=True, stepwise=True, max_order=None)
    # # Print score
    # print("Optimal parameters:", stepwise.get_params())
    # print("AIC score:", stepwise.aic())

    # Results from m=52 auto-arima
    ## Order = (2,0,0)
    ## Seasonal order = (1,1,2,52)
    ## Score = 6964
    ### Score seems high, also lookback is very small


    # Train / test data split
    train = df[df["WeekEnding"] < datetime.datetime.strptime("2015-12-31", "%Y-%m-%d")]
    test = df[df["WeekEnding"] > datetime.datetime.strptime("2015-12-31", "%Y-%m-%d")]

    # Run ARIMA with found parameters
    stepwise = ARIMA(callback=None, disp=0, maxiter=50, method=None, order=(10,1,12), seasonal_order=(2,1,1,52), solver="lbfgs", suppress_warnings=True, transparams=True, trend="c")
    # Fit and predict
    print("Fitting and Predicting...")
    stepwise.fit(train.drop("WeekEnding", axis=1))
    future = stepwise.predict(n_periods=len(test.index))


    # Merge predictions with raw data
    future = pd.DataFrame(future, index=test["WeekEnding"], columns=["Forecast"])
    df = df.set_index("WeekEnding").join(future, how="outer")
    forecast = df.dropna()
    
    # Plot vs actual data
    plt.plot(df)
    plt.xlabel("Date")
    plt.ylabel("Lower 48 Inventory (Bcf)")
    plt.show()

    plt.plot(forecast)
    plt.xlabel("Date")
    plt.ylabel("Lower 48 Inventory (Bcf)")
    plt.show()

    # Much wow, now show how shit the model is
    print(skmetrics.r2_score(forecast["Lower48StocksBcf"], forecast["Forecast"]))