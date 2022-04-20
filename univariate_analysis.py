from dataset import read_dataframe
import pandas as pd
import matplotlib.pyplot as plt

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
