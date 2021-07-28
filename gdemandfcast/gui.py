import os

import streamlit as st

from gdemandfcast.ai import execute


def create():

    st.title("Univariate Forecasting")
    st.subheader("Proof of Concept Demostration")

    gear = st.radio("Model Selection::", ("Auto", "Compare"))
    shift = st.selectbox("Type of Models:", ("ML", "TS", "AUTO"))
    lags = st.slider("Number of Previous Timsteps to Consider:", (1, 5))
    uploaded_train = st.file_uploader("Upload Training Data: ")
    uploaded_test = st.file_uploader("Uploat Testing Data: ")

    df = execute(train, test, lags, gear, shift, speed="fast", charts=False).frm()

    if shift == "auto":
        a = 1
    else:
        a = 4

    for i in range(len(df)):
        st.line_chart(df.loc[:, i : i + a])
        st.bar_chart(df.loc[:, i : i + a])

    return None


# Run Front End on Streamlit
create()
