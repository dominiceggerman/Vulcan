# By Dominic Eggerman

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


# ARIMA modelling
def arima(df, p, d, q):
    # Fit model
    mod = ARIMA(df.set_index("WeekEnding"), order=(p,d,q))
    mod_fit = mod.fit(disp=0)
    print(mod_fit.summary())


# If bitch != side
if __name__ == "__main__":
    # Import data (excel)
    # Note that applying ARIMA to storage levels will likely not produce useful results, as there are many outside factors affecting storage.
    df = pd.read_excel("EIA Weekly May 2019.xlsx", sheet_name="Query Results")
    df = df[["WeekEnding", "Lower48StocksBcf"]]
    df["WeekEnding"] = pd.to_datetime(df["WeekEnding"])

    # Plot storage level
    plt.plot(df["WeekEnding"], df["Lower48StocksBcf"])
    plt.xlabel("Date")
    plt.ylabel("Lower 48 Inventory (Bcf)")
    plt.show()

    # Introduce AutoRegressive Integrated Moving Average (ARIMA)
    ## AutoRegressive : model using dependent relationship between an observation and some number of lagged observations
    ## Integrated : use of differencing of raw observations in order to make time series stationary
    ## Moving Average : model using the dependency between an observation and residual error from a moving average model

    # ARIMA notation is ARIMA(p,d,q) where:
    ## p : the number of lag observations (past values) included in the model
    ## d : the number of times the raw observation are differenced (degrees of differencing)
    ## q : size of the moving average window

    # ?? Assumes that the generated observations is an ARIMA process ??

    # Look at autocorrelation of time series ?? learn WTF this is / how it is made ??
    pd.plotting.autocorrelation_plot(df.set_index("WeekEnding"))
    plt.show()

    # Looking at this plot the positive correlation is 'significant' for the first 8 lags
    # p ~ 8

    # Call arima()
    arima(df, 8, 1, 0)
