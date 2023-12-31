import hopsworks
import datetime
from air_pred.utils import data_preprocessing
import numpy as np
import pandas as pd

FEAURE_GROUP_VERSION = 2
TRAINING_DATASET_VERSION = 1

def update_predictions():
    """ Function that reads from the current predication data frame and clean data dataframe and perform prediction on the newly inserted ones
    """
    project = hopsworks.login()

    fs = project.get_feature_store()
    data_fg = fs.get_feature_group("cleaned_air_quality_data", version=FEAURE_GROUP_VERSION)
    regression_prediction_fg = fs.get_feature_group("predicted_air_quality_regression", version=FEAURE_GROUP_VERSION)

    # Query used instead of a feature view since fv creation fails when this is given as a query
    # Query finds the data points for which predictions are not performed
    query = data_fg.select_all().\
            join(regression_prediction_fg.select([]), on="date_time", join_type = "left").\
            filter(regression_prediction_fg.predicted_femman_pm25 == None)
    try:
        prediction_df = query.sort_values('date_time').drop_duplicates('date_time').reset_index()[["date_time", "date_time_str", "femman_pm25"]]
    except:
        prediction_df = query.read({"use_hive":True}).sort_values('date_time').drop_duplicates('date_time').reset_index()[["date_time", "date_time_str", "femman_pm25"]]
    
    if(len(prediction_df) == 0):
        print("No preditions to preform")
        return None

    fv = fs.get_feature_view("air_qaulity_baseline_fv", version=FEAURE_GROUP_VERSION)
    fv.init_serving(training_dataset_version=TRAINING_DATASET_VERSION)

    prediction_df.date_time = pd.to_datetime(prediction_df.date_time).dt.tz_localize(None)
    start_time=prediction_df.date_time.iloc[0]
    # Work around done for bug fix in which start time when given the value 2023-10-27 00:00:00 was returning a empty df
    if(start_time.hour == 0):
        start_time = start_time - datetime.timedelta(hours=1)
    
    # Getting feature from clean data feature group for which predictions are not performed
    try:
        predication_features = fv.get_batch_data(start_time=start_time)
    except:
        predication_features = fv.get_batch_data(start_time=start_time, read_options={"use_hive":True})
    predication_features.date_time = pd.to_datetime(predication_features.date_time).dt.tz_localize(None)
    # Workaround added since get_batch_data was returning data earlier than start_time at times.
    predication_features = predication_features.sort_values('date_time')[predication_features.date_time>=prediction_df.date_time.iloc[0]]
    predication_features = predication_features.drop(["date_time_str", "date_time"],axis = 1).to_numpy().tolist()

    # Make predications and insert into prediction feature group
    ms = project.get_model_serving()
    my_deployment = ms.get_deployment('aqestimatordeployment')
    predication = np.array([my_deployment.predict(inputs=[feature])["predictions"] for feature in predication_features])

    prediction_df["predicted_femman_pm25"] = predication.squeeze()
    prediction_df["date_time"] = prediction_df["date_time_str"].apply(data_preprocessing.convert_to_datetime)

    regression_prediction_fg.insert(prediction_df, wait=True, write_options={"wait_for_job":True})


if __name__ == "__main__":
    update_predictions()