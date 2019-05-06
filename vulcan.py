# By Dominic Eggerman

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Import data
    df = pd.read_excel("EIA Storage May 2019.xls", "ngsstats 2018 (2013-2017)", header=2)
    
    # Drop last 3 rows
    df.drop(df.tail(3).index, inplace=True)

    # Plot
    plt.plot(range(0, len(df.index)), df["Total Lower 48"])
    plt.show()