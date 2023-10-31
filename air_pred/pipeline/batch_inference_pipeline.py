import hopsworks
import datetime
import joblib
from air_pred.utils import data_preprocessing

project = hopsworks.login(api_key_file="api_key")
fs = project.get_feature_store()
FEAURE_GROUP_VERSION = 2
TRAINING_DATA_VERSION = 1

def create_predication_feature_group():
    predicated_regression_fg = fs.get_or_create_feature_group(name="predicted_air_quality_regression",
                                            version=FEAURE_GROUP_VERSION,
                                            description="Predicted air quality in Gothenburg",
                                            online_enabled=True,
                                            primary_key=["date_time_str"],
                                            event_time='date_time')
    return predicated_regression_fg

def create_predication_feature_view(predicated_regression_fg):
    query = predicated_regression_fg.select_all()
    fv = fs.get_or_create_feature_view(name="predicted_air_quality_regression_fv",
                                       query=query,
                                       version=FEAURE_GROUP_VERSION,
                                        labels=[],
                                       )

def batch_predict(predicated_regression_fg):
    fv = fs.get_feature_view("air_qaulity_baseline_fv", version=FEAURE_GROUP_VERSION)
    fv.init_serving(training_dataset_version=TRAINING_DATA_VERSION)
    try:
        predication_features = fv.get_batch_data()
    except:
        predication_features = predication_features.sort_values('date_time')

    fg = fs.get_feature_group(name="cleaned_air_quality_data", version=FEAURE_GROUP_VERSION)
    try:
        df_data = fg.read()
    except:
        df_data = fg.read({"use_hive":True})

    df_data = df_data.sort_values('date_time')

    predication_features = predication_features.drop(["date_time", "date_time_str"], axis = 1)
    
    mr = project.get_model_registry()
    retrieved_model =  mr.get_best_model("air_quality_estimation_model", metric="Test MSE", direction='min')
    saved_model_dir = retrieved_model.download()

    lr_model = joblib.load(saved_model_dir + "/linear_regression.pkl")
    predication = lr_model.predict(predication_features)

    prediction_df = df_data[["date_time_str", "femman_pm25"]]
    prediction_df["date_time"] = prediction_df["date_time_str"].apply(data_preprocessing.convert_to_datetime)
    prediction_df["predicted_femman_pm25"] = predication.squeeze()

    predicated_regression_fg.insert(prediction_df, wait=True, write_options={"wait_for_job":True})

if __name__ == "__main__":
    predicated_regression_fg = create_predication_feature_group()
    batch_predict(predicated_regression_fg)
    create_predication_feature_view(predicated_regression_fg)