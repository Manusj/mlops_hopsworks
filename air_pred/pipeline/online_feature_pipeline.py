import requests
import pandas as pd
import hopsworks
from air_pred.utils import data_preprocessing
import numpy as np

FEAURE_GROUP_VERSION = 1
project = hopsworks.login(api_key_file="api_key")
fs = project.get_feature_store()

def get_weekly_data():
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
    fg = fs.get_feature_group(name="air_quality_data", version=FEAURE_GROUP_VERSION)

    df = pd.DataFrame(get_weekly_data())
    df = df.replace(r'^\s*$', np.nan, regex=True)

    data = {}
    missing_features = []

    for feature in fg.features:
        if feature.name in df.columns:
            data[feature.name] = data_preprocessing.set_feature_type(df, feature.name, feature.type)
        else:
            missing_features.append([feature.name, feature.type])

    data['date'] = df.date
    data['time'] = df.time
    processed_df = pd.DataFrame(data)

    for missing_features in missing_features:
        feature_name, feature_type = missing_features
        processed_df[feature_name] = np.nan
        processed_df[feature_name] = data_preprocessing.set_feature_type(processed_df, feature_name, feature_type)

    processed_df = data_preprocessing.create_date_time_feature(processed_df).sort_values('date_time')
    fg.insert(processed_df, wait=True, write_options={"wait_for_job":True})
    insert_start_date = processed_df.date_time.iloc[0]
    try:
        new_df = fg.read()
    except:
        new_df = fg.read({"use_hive":True})

    clean_data_fg = fs.get_feature_group(name="cleaned_air_quality_data", version=FEAURE_GROUP_VERSION)
    
    # done so that interpolation can work properly
    cleaned_new_df = data_preprocessing.clean_data(new_df, features=clean_data_fg.features)
    cleaned_new_df['date_time'] = cleaned_new_df['date_time_str'].apply(data_preprocessing.convert_to_datetime)
    cleaned_new_df = cleaned_new_df[cleaned_new_df.date_time >= insert_start_date]
    clean_data_fg.insert(cleaned_new_df, wait=True, write_options={"wait_for_job":True})

if __name__ == "__main__":
    update_feature_groups()


