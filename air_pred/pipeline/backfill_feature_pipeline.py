import pandas as pd
import hopsworks
from air_pred.utils import data_preprocessing
import numpy as np
import great_expectations as ge

project = hopsworks.login(api_key_file="api_key")
fs = project.get_feature_store()

def value_not_null(expectation_suite : ge.core.ExpectationSuite , column_name : list):
    """Function that adds an baseline expectation that the values would not be null in the feature view cleaned_air_quality_data
    Parameters
    ----------
    expectation_suite :  ge.core.ExpectationSuite
        Expectation suit into which the expectations are to be added

    column_name : list
        List of feature names for which the expectations are to be added 
    """
    expectation_suite.add_expectation(ge.core.ExpectationConfiguration(
                                        expectation_type="expect_column_values_to_not_be_null",
                                        kwargs={
                                            "column": column_name,
                                        }
                                        ))

def create_and_fill_baseline_fg(fg_raw_data, initial_df, clean_data_fg, ts_data_fg, expectation_suite_clean_data):
    """ Function that create baseline feature groups based on the csv dataset provided
    """
    fg_raw_data.insert(initial_df, wait=True, write_options={"wait_for_job":True})

    # cleaning read dataframe
    cleneddf = data_preprocessing.clean_data_baseline(initial_df)
    for feature in cleneddf.columns:
        value_not_null(expectation_suite_clean_data, feature)
    
    if clean_data_fg.get_expectation_suite is None:
        clean_data_fg.save_expectation_suite(expectation_suite_clean_data, run_validation=True, validation_ingestion_policy="STRICT")
        ts_data_fg.save_expectation_suite(expectation_suite_clean_data, run_validation=True, validation_ingestion_policy="STRICT")
        
    # Insering cleaned data into cleaned_air_quality_data feature group
    clean_data_fg.insert(cleneddf, wait=True, write_options={"wait_for_job":True})

    tsdf = data_preprocessing.get_time_series_features(clean_data_fg)

    ts_data_fg.insert(tsdf, wait=True, write_options={"wait_for_job":True})

def create_and_fill_iterative_imputer_fg(fg_raw_data, initial_df, clean_data_fg, ts_data_fg, expectation_suite_clean_data, version=2, use_previous_data = True):
    """ Function that create baseline feature groups based on the csv dataset provided
    """
    if use_previous_data:
        try:
            previous_raw_data_fg = fs.get_feature_group(name="air_quality_data", version=version-1)
            try:
                initial_df = previous_raw_data_fg.read()
            except:
                initial_df = previous_raw_data_fg.read({"use_hive":True})
            initial_df["date_time"] = initial_df["date_time_str"].apply(data_preprocessing.convert_to_datetime)
        except:
            pass

    fg_raw_data.insert(initial_df, wait=True, write_options={"wait_for_job":True})

    # cleaning read dataframe
    cleneddf = data_preprocessing.clean_data_IterativeImputer(initial_df)
    for feature in cleneddf.columns:
        value_not_null(expectation_suite_clean_data, feature)
    
    if clean_data_fg.get_expectation_suite is None:
        clean_data_fg.save_expectation_suite(expectation_suite_clean_data, run_validation=True, validation_ingestion_policy="STRICT")
    
    if ts_data_fg.get_expectation_suite is None:
        ts_data_fg.save_expectation_suite(expectation_suite_clean_data, run_validation=True, validation_ingestion_policy="STRICT")
        
    # Insering cleaned data into cleaned_air_quality_data feature group
    clean_data_fg.insert(cleneddf, wait=True, write_options={"wait_for_job":True})

    tsdf = data_preprocessing.get_time_series_features(cleneddf)

    ts_data_fg.insert(tsdf, wait=True, write_options={"wait_for_job":True})

def backfill_air_quality_data(version=1):
    # creating or getting feature group air_quality_data that contains all raw data
    fg_raw_data = fs.get_or_create_feature_group(name="air_quality_data",
                                        version=version,
                                        description="Uncleaned raw data for air quality in Gothenburg",
                                        online_enabled=False,
                                        primary_key=["date_time"],
                                        event_time='date_time')
    
    # reading raw data csv, creating the feature date_time and instering it into feature group
    df = pd.read_csv("air_quality_2023.csv", skipinitialspace = True)
    initial_df = data_preprocessing.create_date_time_feature(df)

    # creating or getting feature group cleaned_air_quality_data that contains cleaned data without any null entries
    clean_data_fg = fs.get_or_create_feature_group(name="cleaned_air_quality_data",
                                            version=version,
                                            description="Cleaned raw data for air quality in Gothenburg",
                                            online_enabled=True,
                                            primary_key=["date_time_str"],
                                            event_time='date_time')
    
    # creating or getting feature group time_series_air_quality_data that contains time series features for femman_pm25
    ts_data_fg = fs.get_or_create_feature_group(name="time_series_air_quality_data",
                                            version=version,
                                            description="Time series features for air quality in Gothenburg",
                                            online_enabled=True,
                                            primary_key=["date_time_str"],
                                            event_time='date_time')
    
    # creating an expectation suit for cleaned_air_quality_data and adding the expecation that all columns are not null 
    expectation_suite_clean_data = ge.core.ExpectationSuite(expectation_suite_name="cleaned_air_quality_fg")
    

    if version == 1:
        create_and_fill_baseline_fg(fg_raw_data=fg_raw_data, initial_df=initial_df, clean_data_fg=clean_data_fg, ts_data_fg=ts_data_fg,  expectation_suite_clean_data=expectation_suite_clean_data)
    elif version == 2:
        create_and_fill_iterative_imputer_fg(fg_raw_data=fg_raw_data, initial_df=initial_df, clean_data_fg=clean_data_fg, ts_data_fg=ts_data_fg, expectation_suite_clean_data=expectation_suite_clean_data, version=2, use_previous_data=True)
if __name__ == "__main__":
    backfill_air_quality_data(version=2)