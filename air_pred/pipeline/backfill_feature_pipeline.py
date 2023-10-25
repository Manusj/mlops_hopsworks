import pandas as pd
import hopsworks
from air_pred.utils import data_preprocessing
import numpy as np
import great_expectations as ge

FEAURE_GROUP_VERSION = 1
project = hopsworks.login(api_key_file="api_key")
fs = project.get_feature_store()

def value_not_null(expectation_suite, column_name):
    expectation_suite.add_expectation(ge.core.ExpectationConfiguration(
                                        expectation_type="expect_column_values_to_not_be_null",
                                        kwargs={
                                            "column": column_name,
                                        }
                                        ))

def backfill_air_quality_data():
    fg = fs.get_or_create_feature_group(name="air_quality_data",
                                        version=FEAURE_GROUP_VERSION,
                                        description="Uncleaned raw data for air quality in Gothenburg",
                                        online_enabled=False,
                                        primary_key=["date_time"],
                                        event_time='date_time')
    df = pd.read_csv("air_quality_2023.csv", skipinitialspace = True)
    processed_df = data_preprocessing.create_date_time_feature(df)
    fg.insert(processed_df, wait=True, write_options={"wait_for_job":True})

    clean_data_fg = fs.get_or_create_feature_group(name="cleaned_air_quality_data",
                                            version=FEAURE_GROUP_VERSION,
                                            description="Cleaned raw data for air quality in Gothenburg",
                                            online_enabled=True,
                                            primary_key=["date_time_str"],
                                            event_time='date_time')
    expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="cleaned_air_quality_fg")
    for feature in fg.features:
        value_not_null(expectation_suite, feature.name)
    try:
        fg.save_expectation_suite(expectation_suite, run_validation=True, validation_ingestion_policy="STRICT")
    except:
        pass
    # Must do backfill also
    cleneddf = data_preprocessing.clean_data(processed_df)
    clean_data_fg.insert(cleneddf, wait=True, write_options={"wait_for_job":True})
    return fg

if __name__ == "__main__":
    backfill_air_quality_data()