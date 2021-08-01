import pandas as pd
import streamlit as st

from gdemandfcast.ai import automate, execute

st.title("Time Series Forecasting")
st.subheader("Proof of Concept Demostration")
st.markdown("***")
st.markdown("***")
lags = st.slider("Number of Previous Timsteps to Consider:", 1, 5, 3, 1)
gear = st.radio("Model Selection::", ("Compare", "Auto"))
shift = st.selectbox("Type of Models:", ("ML", "TS", "DL"))
speed = st.radio("Fast or Slow ML Tuning:", ("Fast", "Slow"))
train = st.file_uploader("Upload Training Data: ")
test = st.file_uploader("Uploat Testing Data: ")
st.markdown("***")
gear = gear.lower()
shift = shift.lower()

if train is not None and test is not None:

    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # st.write(train)

        train_X, train_y, test_X, test_y = execute(train, test, lags).get()
        df, percentage_accurate = automate(
            train_X, train_y, test_X, test_y, gear, shift, speed
        ).run()

        st.line_chart(df)
        st.bar_chart(df)

else:

    submit = st.button("Run Demo")
    if submit:

        train = "./data/Train Data.xlsx"
        test = "./data/Test Data.xlsx"
        df_train = pd.read_excel(train).fillna(0)
        df_test = pd.read_excel(test).fillna(0)

        for col in df_train.columns:

            train = df_train[[col]].reset_index(drop=True)
            test = df_test[[col]].reset_index(drop=True)
            # st.write(train)

            train_X, train_y, test_X, test_y = execute(train, test, lags).get()
            df, percentage_accurate = automate(
                train_X, train_y, test_X, test_y, gear, shift, speed
            ).run()

            st.line_chart(df)
            st.bar_chart(df)
