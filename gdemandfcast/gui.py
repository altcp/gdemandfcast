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


def get(train, test, lags):
    train1 = pd.read_excel(train)
    test1 = pd.read_excel(test)
    train2 = train1.fillna(0)
    test2 = test1.fillna(0)
    train_X, train_y, test_X, test_y = execute(train2, test2, lags).get()

    return train_X, train_y, test_X, test_y


if train is not None and test is not None:

    train_X, train_y, test_X, test_y = get(train, test, lags)
    df, percentage_accurate = automate(
        train_X, train_y, test_X, test_y, gear, shift, speed
    ).run()
    if gear == "auto":
        a = 1
    else:
        a = 4

    for i in range(len(df)):
        st.line_chart(df.loc[:, i : i + a])
        st.bar_chart(df.loc[:, i : i + a])

else:

    submit = st.button("Run Demo")
    if submit:

        train = "./data/Train Data.xlsx"
        test = "./data/Test Data.xlsx"
        train_X, train_y, test_X, test_y = get(train, test, lags)
        df, percentage_accurate = automate(
            train_X, train_y, test_X, test_y, gear, shift, speed
        ).run()
        st.write("Demostration Based on Seen Data.")

        print(df)

        if gear == "auto":
            a = 1
        else:
            a = 4

        for i in range(len(df)):
            st.line_chart(df.loc[:, i : i + a])
            st.bar_chart(df.loc[:, i : i + a])
