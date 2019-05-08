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

    # Run ARIMA with found parameters
    stepwise = ARIMA(callback=None, disp=0, maxiter=50, method=None, order=(8,1,12), seasonal_order=(2,1,1,52), solver="lbfgs", suppress_warnings=True, transparams=True, trend="c")
    # Fit and predict
    print("Fitting and Predicting...")
    stepwise.fit(df.drop("WeekEnding", axis=1))
    future = stepwise.predict(n_periods=52)


    # Merge predictions with raw data
    year_start = datetime.datetime.strptime("2018-12-28", "%Y-%m-%d") + datetime.timedelta(days=7)
    report_week = [year_start + datetime.timedelta(days=7*i) for i in range(52)]
    future = pd.DataFrame(future, index=report_week, columns=["Forecast"])
    df = df.set_index("WeekEnding").join(future, how="outer")
    
    # Plot vs actual data
    plt.plot(df)
    plt.xlabel("Date")
    plt.ylabel("Lower 48 Inventory (Bcf)")
    plt.show()

    # Much wow, now show how shit the model is