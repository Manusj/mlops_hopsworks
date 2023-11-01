import hopsworks
import os

project = hopsworks.login(api_key_file="api_key")
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

uploaded_file_path = dataset_api.upload("predictor.py", "Models", overwrite=True)
predictor_script_path = os.path.join("/Projects", project.name, uploaded_file_path)

def deploy_linear_regression_baseline():
    mr = project.get_model_registry()
    model = mr.get_best_model("air_quality_estimation_model", metric="Test MSE", direction='min')

    # Give it any name you want
    deployment = model.deploy(
        name="aqestimatordeployment", 
        serving_tool="KSERVE",
        script_file=predictor_script_path
    )
    
    deployment.start()

    model = mr.get_best_model("air_quality_time_series_model", metric="Test MSE", direction='min')

    # Give it any name you want
    deployment = model.deploy(
        name="aqtsdeployment", 
        serving_tool="KSERVE",
        script_file=predictor_script_path
    )
    
    deployment.start()

    return deployment

if __name__ == "__main__":
    deploy_linear_regression_baseline()