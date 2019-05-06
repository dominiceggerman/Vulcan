# By Dominic Eggerman

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Import data (excel)
    ex = pd.read_excel("EIA Storage May 2019.xls", sheetname=None, header=2)
    
    # Loop through sheets
    df = pd.Dataframe()
    for key in ex:
        ex[key].drop(df.tail(3).index, inplace=True)
        df = df.append(ex[key]["Total Lower 48"])
    df.reset_index(inplace=True)
    print(df.head())

    # Plot
    plt.plot(range(0, len(df.index)), df["Total Lower 48"])
    plt.show()