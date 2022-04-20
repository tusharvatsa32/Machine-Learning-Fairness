import pandas as pd
import sys
sys.path.append('./Data')
def read_dataframe() -> pd.DataFrame:
    """
    A function to read the data and return it as a csv DataFrame

    Returns
    -------
    pd.DataFrame
        a pandas DataFrame object
    """
    return pd.read_csv('./Data/german_credit_data.csv', index_col = 0)



def check_head(dataframe : pd.DataFrame) -> pd.DataFrame:
    """
    A function to check the top 5 values of the dataset.

    Parameters
    ----------
    pd.DataFrame
        a pandas DataFrame object

    Returns
    ---------
    pd.DataFrame
        top 5 values of the DataFrame object
    """
    return dataframe.head()

def check_info(dataframe : pd.DataFrame) -> pd.DataFrame:
    """
    A function to check the type of features(data) present in the dataset.

    Parameters
    ----------
    pd.DataFrame
        a pandas DataFrame object

    Returns
    ---------
    pd.DataFrame
        Columns present and their datatypes
    """
    return dataframe.info()

def check_unique_values(dataframe : pd.DataFrame) -> pd.DataFrame:
    """
    A function to check the number of unique values in each feature in the dataset.

    Parameters
    ----------
    pd.DataFrame
        a pandas DataFrame object

    Returns
    ---------
    pd.DataFrame
        Columns present and the number of unique values in them
    """
    return dataframe.nunique()

def describe_dataset(dataframe : pd.DataFrame) -> pd.DataFrame:
    """
    A function to describe the statistical properties in the dataset.

    Parameters
    ----------
    pd.DataFrame
        a pandas DataFrame object

    Returns
    ---------
    pd.DataFrame
        Columns with numerical values and their statistical properties.
    """
    return dataframe.describe()


if __name__ == "__main__":
    df = read_dataframe()
    print(df)
    print(check_head(df))
    print(check_info(df))
    print(check_unique_values(df))
    print(describe_dataset(df))

