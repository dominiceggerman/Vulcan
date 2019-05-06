# By Dominic Eggerman

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Import data (excel)
    df = pd.read_excel("EIA Weekly May 2019.xlsx", sheet_name="Query Results")

    # Plot
    plt.plot(df["WeekEnding"], df["Lower48StocksBcf"])
    plt.show()