# By Dominic Eggerman

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Import data (excel)
    ex = pd.read_excel("EIA Storage May 2019.xls", sheet_name=None, header=2)
    
    # Loop through sheets and load data
    df = pd.DataFrame()
    for key in ex:
        print('1')
        df = pd.concat([df, ex[key]["Total Lower 48"].drop(df.tail(3).index)])
    # df.reset_index(inplace=True)
    print(df)

    # Plot
    # plt.plot(range(0, len(df.index)), df["Total Lower 48"])
    # plt.show()