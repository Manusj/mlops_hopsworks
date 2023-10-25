import pandas as pd
import datetime as dt
from hsfs.feature import Feature

def convert_to_datetime(date_str : str):
    """Function used to convert a column into data time that contains time value 24:00 to datetime format of pandas
    Parameters
    ----------
    date_str : string
        Date and time string in the format %Y-%m-%d %H:%M

    Returns
    -------
    datetime64
        converted date time string
    """

    # directly use pd.to_datetime function if the hour value is not 24
    if date_str[11:13] != '24':
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M')

    # if hour value is 24 then convert it to 00 and add 1 to the day to show it is in the next day
    date_str = date_str[0:11] + '00' + date_str[13:]
    return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M') + \
           dt.timedelta(days=1)

def create_date_time_feature(input_df : pd.DataFrame) -> pd.DataFrame:
    """Function that takes in a dataframe that contains seperate colums for date, time 
        and returns a dataframe with a single column for date and time having the correct pandas Datatime format.

    Parameters
    ----------
    input_df : pd.DataFrame
        input dataframe that contains the time and date in seperate columns

    Returns
    -------
    pd.Dataframe
        converted dataframe that contains the date and time in a single columns with correct format
    """
    input_df.columns = input_df.columns.str.lower().str.strip()
    input_df['date_time_str'] = input_df['date'] +' ' + input_df['time'].str.split('+', expand=True)[0]
    input_df['date_time'] =  input_df['date_time_str'].apply(convert_to_datetime)
    return input_df.drop(['date', 'time'], axis =1)

def remove_nan_features(df : pd.DataFrame) -> pd.DataFrame:
    """Function to identify features that have more than 50% of nan values and remove them from dataframe

    Parameters
    ----------
    input_df : pd.DataFrame
        input dataframe that contains the time and date in seperate columns

    Returns
    -------
    pd.Dataframe
        converted dataframe that does not contain features with more than 50% nan data
    """
    noisy_features = df.columns[df.isna().sum()/len(df) * 100 > 50].tolist()
    return df.drop(noisy_features, axis=1)

def clean_data(df: pd.DataFrame, features : list = None ) -> pd.DataFrame:
    """Function to clean dataframe of nan data a simple interplotation stratagy is used for now

    Parameters
    ----------
    input_df : pd.DataFrame
        input dataframe that contains the time and date in seperate columns
    list : List of Feature object 
        List of features objects that contains features that must be present in returned dataframe. 
        Is None if a new feature groups is being created and no current schema exists
    Returns
    -------
    pd.Dataframe
        converted dataframe that does not contain features with nan data
    """
    if features is None:
        df = remove_nan_features(df)
    else:
        colums = [feature.name for feature in features]
        df = df[colums]
    return df.sort_values('date_time').interpolate()


def set_feature_type(df, feature_name, feature_type):
    if feature_type == "double":
        return df[feature_name].astype(float)
    elif feature_type == "bigint":
        return df[feature_name].astype(int)
    else:
        return df[feature_name].astype(str)  



