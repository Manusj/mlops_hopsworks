import requests
import pandas as pd
import hopsworks
from air_pred.utils import data_preprocessing
import numpy as np
import datetime

FEAURE_GROUP_VERSION = 2
project = hopsworks.login()
fs = project.get_feature_store()

def get_weekly_data():
    """Function that reads weekly data from gothenburg open data portal API
    """
    offset = 0
    results = []
    while(True):
        weekly_data = requests.get(f'https://catalog.goteborg.se/rowstore/dataset/85ae9601-5258-442b-bc65-d74549c0cf8a/json?_offset={offset}&_limit=100').json()
        limit = weekly_data["limit"]
        total_results = weekly_data["resultCount"]
        results += weekly_data["results"]
        if(total_results > offset+limit):
            offset += limit
        else:
            break
    return results


def update_feature_groups():
    """ Function that reads the data from the open data portal and adds it to the feature groups
    """
    # Getting feature groups for cleaned data and time series features
    fg = fs.get_feature_group(name="air_quality_data", version=FEAURE_GROUP_VERSION)
    tf_fg = fs.get_feature_group(name="time_series_air_quality_data", version=FEAURE_GROUP_VERSION)

    # cleaning data read from the API to remove white spaces and replace with nan's like hostorical csv file
    df = pd.DataFrame(get_weekly_data())
    df = df.replace(r'^\s*$', np.nan, regex=True)

    data = {}
    missing_features = []

    # Finding features missing in the newly recived api call
    for feature in fg.features:
        if feature.name in df.columns:
            data[feature.name] = data_preprocessing.set_feature_type(df, feature.name, feature.type)
        else:
            missing_features.append([feature.name, feature.type])

    data['date'] = df.date
    data['time'] = df.time
    processed_df = pd.DataFrame(data)
    # setting missing features as nan
    for missing_features in missing_features:
        feature_name, feature_type = missing_features
        processed_df[feature_name] = np.nan
        processed_df[feature_name] = data_preprocessing.set_feature_type(processed_df, feature_name, feature_type)

    # doing same prepossing steps and instering raw data to raw data feature group
    processed_df = data_preprocessing.create_date_time_feature(processed_df).sort_values('date_time')
    fg.insert(processed_df, wait=True, write_options={"wait_for_job":True})
    insert_start_date = processed_df.date_time.iloc[0]

    #using inserted raw data to read and create the cleaned data. Complete raw data read since imputation of missing values would be more accurate
    try:
        new_df = fg.read()
    except:
        new_df = fg.read({"use_hive":True})

    clean_data_fg = fs.get_feature_group(name="cleaned_air_quality_data", version=FEAURE_GROUP_VERSION)
    
    # cleaning data and inserting into cleaned data feature group
    cleaned_new_df_full = data_preprocessing.clean_data_IterativeImputer(new_df, features=clean_data_fg.features)
    cleaned_new_df_full['date_time'] = cleaned_new_df_full['date_time_str'].apply(data_preprocessing.convert_to_datetime)
    cleaned_new_df = cleaned_new_df_full[cleaned_new_df_full.date_time >= insert_start_date]
    clean_data_fg.insert(cleaned_new_df, wait=True, write_options={"wait_for_job":True})

    ## Cerating features for time series. 
    start_time = cleaned_new_df_full.date_time.iloc[-1]
    palceholder = pd.DataFrame(columns=cleaned_new_df_full.columns)
    # Setting future 24 hour value for femme_pm25 which is not know now as -1. This will be overwritten when data is avaiable
    for i in range(1,25):
        future_time = start_time+ datetime.timedelta(hours = i)
        data = [future_time if col == "date_time" else -1 for col in cleaned_new_df_full.columns]
        palceholder.loc[len(palceholder)] = data 

    # Creating features and inserting to feature group
    palceholder.date_time_str = palceholder.date_time.astype(str)
    cleaned_new_df_full = pd.concat([cleaned_new_df_full, palceholder]).sort_values("date_time").reset_index()
    tf_df = data_preprocessing.get_time_series_features(cleaned_new_df_full)
    tf_fg.insert(tf_df, wait=True, write_options={"wait_for_job":True})


if __name__ == "__main__":
    update_feature_groups()


