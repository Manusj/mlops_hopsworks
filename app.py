import streamlit as st
import hopsworks
import datetime 
import plotly.graph_objects as go


st.header('Air Quality Estimation - Gothenburg', divider="blue")

FEATURE_VERSION = 1

project = hopsworks.login(api_key_file="api_key")
fs = project.get_feature_store()
fv = fs.get_feature_view(name="predicted_air_quality_regression_fv", version=FEATURE_VERSION)


option = st.selectbox(
   "Estimation Period",
   ("24 hours", "1 Week", "1 Month"),
   index=0)


def get_predications(_fv, options):
    if(options == "24 hours"):
        time_delta = datetime.timedelta(days=1)
    elif(options == "1 Week"):
        time_delta = datetime.timedelta(days=7)
    elif(option):
        time_delta = datetime.timedelta(days=30)
    start_time = datetime.datetime.now() - time_delta
    end_time = datetime.datetime.now()
    df = fv.get_batch_data(start_time=start_time, end_time=end_time)
    return df

@st.cache_data
def get_predication_plot(df, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.date_time, y=df.femman_pm25,
                        mode='markers',
                        name='actual'))
    fig.add_trace(go.Scatter(x=df.date_time, y=df.predicted_femman_pm25,
                        mode='markers',
                        name='predicted'))
    fig.update_layout(xaxis_title="Date - Time", yaxis_title="PM 25", title = title)
    return fig

st.write(get_predication_plot(get_predications(fv,option), f'{option} Predication'))
