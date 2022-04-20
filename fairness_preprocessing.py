from dataset import read_dataframe, check_head
import pandas as pd
import numpy as np
def preprocessing(df : pd.DataFrame) -> pd.DataFrame:
    """
    Processed data for model training:
    Deals with missing values
    Normalize the values for numerical features
    Encodes the values for categorical features

    Parameters
    ----------
    df
        The dataframe corresponding to the german credit dataset

    Returns
    -------
    pd.DataFrame
        The dataframe after preprocessing

    """
    df_credit = df

    #Change Risk column (target variable) into numerical with 0 and 1 values
    df_y = df[['Risk']]
    df_y['Risk'] = np.where(df_y.loc[ : , 'Risk']=='good', 0, 1)

    #Forming the dataframe without the target variable
    df_credit = df_credit.drop(["Risk"], axis=1)

    #Dealing with missing values
    df_credit['Saving accounts'].fillna('little', inplace = True)
    df_credit['Checking account'].fillna('no_inf', inplace=True)

    #Normalizing the values for numerical features
    df_credit = log_transform(df_credit, ['Credit amount', 'Duration', 'Age'])

    #Encodes the categorical features
    df_credit_T = one_hot_encoder(df_credit, False)

    #Concatenating the predictor variables with the labels to form training data
    data_train = pd.concat([df_credit_T, df_y], axis=1)
    return data_train



def one_hot_encoder(df : pd.DataFrame, nan_as_category : bool) -> pd.DataFrame:
    """
    Convert all object and category variables into numerical values by one hot encoding.

    Parameters
    ----------
    df
        The dataframe after normalizing

    nan_as_category
        If we want to convert nan as a separate category

    Returns
    -------
    pd.DataFrame
        The modified dataframe after encoding the categorical values
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns = categorical_columns, dummy_na = nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df


def log_transform(data : pd.DataFrame, to_log : list) -> pd.DataFrame:
    """
    Calculates the log transform for the features.

    Parameters
    ----------
    data
        The dataframe after some amount of preprocessing

    to_log
        A list of features from the dataset

    Returns
    -------
    pd.DataFrame
        The log transformed features of the dataset
    """
    X = data.copy()
    for item in to_log:
        # Add 1 to the data to prevent infinity values
        X[item] = np.log(1+X[item])
    return X
