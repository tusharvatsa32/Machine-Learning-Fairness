from dataset import read_dataframe
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

def plotObjColumnDistribution(df : pd.DataFrame, nGraphShown : int, nGraphPerRow : int):
    """
    A function which plots the count of values for features :
    'Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk'

    Parameters
    ----------
    df
        The dataframe corresponding to the german credit dataset

    nGraphShown
        The number of graphs to be shown. In this case : 6

    nGraphPerRow
        The number of graphs in each row.

    """
    print(f'=== Distribution of features with object values ===')

    df = df[[col for col in df.select_dtypes(["object", "category"])]] # pick columns with object type

    warnings.filterwarnings('ignore')
    cmap=sns.color_palette('Blues_r')

    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 5 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i, col in enumerate(columnNames):
        # Plot distribution
        plt.subplot(int(nGraphRow), nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        valueCounts = columnDf.value_counts()
        valueCounts.plot.bar()

        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')

    # Show the plot
    plt.tight_layout()
    plt.show()

def plotNumColumnDistribution(df : pd.DataFrame, nGraphPerRow : int):
    """
    A function to visualize the numerical variables and their distribution.

    Parameters
    ----------
    df
        The dataframe corresponding to the german credit dataset

    nGraphPerRow
        The number of graphs in each row

    """
    print(f'=== Distribution of features with number values ===')

    df = df[[col for col in df.select_dtypes(np.number)]] # pick columns with bumber type

    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 3 * nGraphRow))

    for i, col in enumerate(columnNames):
        # Plot distribution
        plt.subplot(int(nGraphRow), nGraphPerRow, i + 1)
        sns.distplot(df[col], color='blue')
        plt.title(f'Distribution of {col}')
    # Show the plot
    plt.tight_layout()
    plt.show()
