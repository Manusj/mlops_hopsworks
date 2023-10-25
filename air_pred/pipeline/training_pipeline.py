import hopsworks
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

FEAURE_GROUP_VERSION = 1

def create_featureView():
    project = hopsworks.login(api_key_file="api_key")
    fs = project.get_feature_store()
    fg = fs.get_feature_group("cleaned_air_quality_data")
    query = fg.select_all()
    transformations = {feature.name: fs.get_transformation_function(name="min_max_scaler") \
                       for feature in fg.features if feature.name not in ['femman_pm25', "date_time", "date_time_str"]}
    fv = fs.get_or_create_feature_view(name="air_qaulity_baseline_fv",
                                       query=query,
                                       version=1,
                                        labels=['femman_pm25'],
                                       transformation_functions=transformations
                                       )
    try:
        df = fg.read()
    except:
        df = fg.read({"use_hive":True})
    df = df.sort_values('date_time')
    train_start_date = df.date_time.iloc[0]
    train_end_date =  df.date_time.iloc[int(0.8*len(df))]
    test_start_date = df.date_time.iloc[int(0.8*len(df))]
    test_end_date = df.date_time.iloc[len(df)-1]
    version, job = fv.create_train_test_split(train_start=train_start_date, train_end=train_end_date,
                                              test_start=test_start_date, test_end=test_end_date,
                                              data_format="csv", description="Basline train test split",write_options={"wait_for_job":True})

    return fv

    


def train_model():
    project = hopsworks.login(api_key_file="api_key")
    fs = project.get_feature_store()
    fv = fs.get_feature_view("air_qaulity_baseline_fv", version=1)
    trainX, testX, trainY, testY = fv.get_train_test_split(training_dataset_version=1)
    trainX = trainX.drop(["date_time", "date_time_str"], axis=1)
    testX = testX.drop(["date_time", "date_time_str"], axis=1)
    reg = LinearRegression().fit(trainX, trainY)
    train_score = reg.score(trainX, trainY)
    test_score = reg.score(testX, testY)
    print(f'Train Score : {train_score}, Test Score : {test_score}')

    mr = project.get_model_registry()
    joblib.dump(reg, './models/linear_regression.pkl')

    input_schema = Schema(trainX)
    output_schema = Schema(trainY)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    model_schema.to_dict()

    test_example = testX.iloc[-1]
    
    model = mr.python.create_model(name="linear_regression_baseline",
                                   metrics={"Train Score":train_score, "Test Score": test_score},
                                   description="Basline linear regssion model",
                                   input_example=test_example,
                                   model_schema=model_schema)
    model.save('./models/linear_regression.pkl')

if __name__ == "__main__":
    create_featureView()
    train_model()
