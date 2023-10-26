import hopsworks
import datetime
from air_pred.utils import data_preprocessing
import numpy as np
import pandas as pd

def update_predictions():
    project = hopsworks.login(api_key_file="api_key")

    fs = project.get_feature_store()
    data_fg = fs.get_feature_group("cleaned_air_quality_data")
    regression_prediction_fg = fs.get_feature_group("predicted_air_quality_regression")

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

    fv = fs.get_feature_view("air_qaulity_baseline_fv", version=1)
    fv.init_serving(training_dataset_version=1)

    try:
        predication_features = fv.get_batch_data(start_time=prediction_df.date_time.iloc[0], end_time = prediction_df.date_time.iloc[-1]+datetime.timedelta(hours=1))
    except:
        predication_features = fv.get_batch_data(start_time=prediction_df.date_time.iloc[0], end_time = prediction_df.date_time.iloc[-1]+datetime.timedelta(hours=1), read_options={"use_hive":True})
    predication_features.date_time = pd.to_datetime(predication_features.date_time).dt.tz_localize(None)
    predication_features = predication_features.sort_values('date_time')[predication_features.date_time>=prediction_df.date_time.iloc[0]]
    predication_features = predication_features.drop(["date_time_str", "date_time"],axis = 1).to_numpy().tolist()

    ms = project.get_model_serving()
    my_deployment = ms.get_deployment('lrbasedeployment')
    predication = np.array([my_deployment.predict(inputs=[feature])["predictions"] for feature in predication_features])

    prediction_df["predicted_femman_pm25"] = predication.squeeze()
    prediction_df["date_time"] = prediction_df["date_time_str"].apply(data_preprocessing.convert_to_datetime)
    regression_prediction_fg.insert(prediction_df, wait=True, write_options={"wait_for_job":True})


if __name__ == "__main__":
    update_predictions()