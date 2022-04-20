import plotly.offline as py
from dataset import read_dataframe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from collections import Counter

def plotRiskNumColumnDistribution(df : pd.DataFrame, nGraphPerRow : int):
    """
    This function plots the count of certain features against Risk.
    Those features are : Age, Job, Credit amount, Duration
    Risk is defined as either Yes or No
    Yes means the risk of lending to the borrower is high
    No means the risk of lending to the borrower is low.

    Parameters
    ----------
    df
        The dataframe corresponding to the german credit dataset

    nGraphPerRow
        The number of graphs in each row

    """
    df2 = df[[col for col in df.select_dtypes(np.number)]] # pick columns with number type

    nRow, nCol = df2.shape
    columnNames = list(df2)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    fig, axes = plt.subplots(int(nGraphRow), nGraphPerRow, figsize=(15, 8))
    axes = axes.ravel()

    for ax in axes:
        ax.set_axis_off()

    for i in range(len(columnNames)):
        axes[i].hist(df[df["Risk"]=="good"][columnNames[i]], alpha=0.5, color='blue',  label='No')
        axes[i].hist(df[df["Risk"]=="bad"][columnNames[i]], alpha=0.5, color='red',  label='Yes')

        axes[i].legend(prop={'size': 10})
        axes[i].set_title(str(columnNames[i]))
        axes[i].set_axis_on()

    fig.tight_layout()
    plt.show()


def plotRiskObjColumnDistribution(df : pd.DataFrame, nGraphPerRow : int):
    """
    This function plots the count of categorical features against Risk.
    Those features are : Sex, Housing, Saving accounts, Checking account, Purpose
    Risk is defined as either Yes or No
    Yes means the risk of lending to the borrower is high
    No means the risk of lending to the borrower is low.

    Parameters
    ----------
    df
        The dataframe corresponding to the german credit dataset

    nGraphPerRow
        The number of graphs in each row

    """
    print(f'=== Distribution of features with object values ===')
    df2 = df[[col for col in df.select_dtypes(["object", "category"])]] # pick columns with object type
    nRow, nCol = df2.shape


    columnNames = list(df2)


    columnNames = [item for item in columnNames if item != "Risk"]


    nGraphRow = int(len(columnNames)/ nGraphPerRow) + 1

    figsize = (6 * nGraphPerRow, 3 * nGraphRow)
    for i, col in enumerate(columnNames):
        df_pct = df2.groupby([columnNames[i],'Risk'])['Risk'].count()/df2.groupby([columnNames[i]])['Risk'].count()

        df_pct.unstack().plot.bar(stacked=True)
        plt.ylabel('counts percent %')
        plt.title(f'Risk distribution with {columnNames[i]}')

    plt.tight_layout()
    plt.show()



