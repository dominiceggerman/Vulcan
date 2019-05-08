# By Dominic Eggerman

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima, ARIMA


# If bitch != side
if __name__ == "__main__":
    # Import data (excel)
    # Note that applying ARIMA to storage levels will likely not produce useful results, as there are many outside factors affecting storage
    df = pd.read_excel("EIA Weekly May 2019.xlsx", sheet_name="Query Results")
    df = df[["WeekEnding", "Lower48StocksBcf"]]
    df["WeekEnding"] = pd.to_datetime(df["WeekEnding"])

    # # Plot storage level
    # plt.plot(df)
    # plt.xlabel("Date")
    # plt.ylabel("Lower 48 Inventory (Bcf)")
    # plt.show()

    # # Decompose the time series using seasonal_decompose ?? Residuals ??
    # decomp = seasonal_decompose(df.set_index("WeekEnding"), model="multiplicative")
    # decomp.plot()
    # plt.show()
    # # Residuals seem a touch high, but multiplicative model seems to fit well ??

    # Introduce AutoRegressive Integrated Moving Average (ARIMA)
    ## AutoRegressive : model using dependent relationship between an observation and some number of lagged observations
    ## Integrated : use of differencing of raw observations in order to make time series stationary
    ## Moving Average : model using the dependency between an observation and residual error from a moving average model

    # ARIMA notation is ARIMA(p,d,q) where:
    ## p : the number of lag observations (past values) included in the model
    ## d : the number of times the raw observation are differenced (degrees of differencing)
    ## q : size of the moving average window

    # We can add a seasonal element to the ARIMA model (SARIMA), with elements SARIMA(P,D,Q) which explain the seasonal component

    # # To find values of p, d, and q we can use autocorrelation, correlation, domain experience, etc. ??
    # # Auto-correlation. How does this work ??
    # pd.plotting.autocorrelation_plot(df.set_index("WeekEnding"))
    # plt.show()

    # Looking at this plot the positive correlation is 'significant' for the first 8 lags
    # p ~ 8

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

    # Order = (2,0,0)
    # Seasonal order = (1,1,2,52)
    # Score = 6964
    ## Score seems high, also lookback is very small

    # Train / test split ?? Experiment with splitting
    train = df[df["WeekEnding"] < datetime.datetime.strptime("2017-12-31", "%Y-%m-%d")]
    test = df[df["WeekEnding"] > datetime.datetime.strptime("2017-12-31", "%Y-%m-%d")]

    # Run ARIMA with found parameters
    stepwise = ARIMA(callback=None, disp=0, maxiter=50, method=None, order=(2,0,2), seasonal_order=(2,1,0,12), solver="lbfgs", suppress_warnings=True, transparams=True, trend="c")
    # Fit and predict
    print("Fitting and Predicting...")
    stepwise.fit(train.drop("WeekEnding", axis=1))
    future = stepwise.predict(n_periods=len(test.index))

    # Merge data
    future = pd.DataFrame(future, index=test["WeekEnding"], columns=["Forecast"])
    df = df.set_index("WeekEnding").join(future, how="outer")
    
    # Plot vs actual data
    plt.plot(df)
    plt.xlabel("Date")
    plt.ylabel("Lower 48 Inventory (Bcf)")
    plt.show()

    plt.plot(df.tail(len(test.index)))
    plt.xlabel("Date")
    plt.ylabel("Lower 48 Inventory (Bcf)")
    plt.show()

    # Much wow, now show how shit the model is