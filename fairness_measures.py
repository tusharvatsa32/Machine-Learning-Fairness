import pandas as pd

def anti_classification_age(df : pd.DataFrame) -> pd.DataFrame:
    """
    Remove age feature from the dataset.

    Parameters
    ----------
    df
        The german credit dataset without labels

    Returns
    -------
    pd.DataFrame
        The dataset after removing the age feature
    """
    df_credit_anti_age = df.drop(["Age"], axis=1)
    return df_credit_anti_age

def anti_classification_gender(df : pd.DataFrame) -> pd.DataFrame:
    """
    Remove gender feature from the dataset.

    Parameters
    ----------
    df
        The german credit dataset without labels

    Returns
    -------
    pd.DataFrame
        The dataset after removing the gender feature
    """
    df_credit_anti_gender = df.drop(["Sex_female","Sex_male"], axis=1)
    return df_credit_anti_gender

def group_fairness(df : pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the demographic parity for age and gender.

    Parameters
    ----------
    df
        The german credit dataset without labels

    Returns
    -------
    confusion matrix
        True positive, False positive, True Negative, False Negative
    age_up
        For ages above the median age
    age_down
        For ages below the median age
    class_female
        For female class
    class_male
        For male class
    """
    age_median=df["Age"].median()

    age_up=len(df[(df['predictions']==1) & (df['Age']>=age_median)])/len(df[(df['Age']>=age_median)])
    age_down=len(df[(df['predictions']==1) & (df['Age']<age_median)])/len(df[(df['Age']<age_median)])

    class_female = len(df[(df['predictions']==1) & (df['Sex_female']==1)])/len(df[(df['Sex_female']==1)])
    class_male = len(df[(df['predictions']==1) & (df['Sex_male']==1)])/len(df[(df['Sex_male']==1)])

    confusion_matrix = pd.crosstab(df['predictions']==1,df['Sex_male']==1, rownames=['Actual'], colnames=['Predicted'])
    return confusion_matrix, (age_up, age_down), (class_female, class_male)


def separation(df_pred_sep : pd.DataFrame) -> pd.DataFrame:

    """
    Parameters
    ----------
    df
        The german credit dataset without labels

    Returns
    -------
    confusion matrix
        True positive, False positive, True Negative, False Negative
    age_up
        For ages above the median age
    age_down
        For ages below the median age
    class_female
        For female class
    class_male
        For male class
    """

    age_median=df_pred_sep["Age"].median()

    age_up=len(df_pred_sep[(df_pred_sep['predictions']==1) & (df_pred_sep['Age']>=age_median) & (df_pred_sep['Risk']==0)])/len(df_pred_sep[(df_pred_sep['Age']>=age_median)& (df_pred_sep['Risk']==0)])
    age_down=len(df_pred_sep[(df_pred_sep['predictions']==1) & (df_pred_sep['Age']<age_median) & (df_pred_sep['Risk']==0)])/len(df_pred_sep[(df_pred_sep['Age']<age_median) & (df_pred_sep['Risk']==0)])

    class_female=len(df_pred_sep[(df_pred_sep['predictions']==1) & (df_pred_sep['Sex_female']==1) & (df_pred_sep['Risk']==0)])/len(df_pred_sep[(df_pred_sep['Sex_female']==1) & (df_pred_sep['Risk']==0)])
    class_male=len(df_pred_sep[(df_pred_sep['predictions']==1) & (df_pred_sep['Sex_male']==1) & (df_pred_sep['Risk']==0)])/len(df_pred_sep[(df_pred_sep['Sex_male']==1) & (df_pred_sep['Risk']==0)])

    return (age_up, age_down), (class_female, class_male)