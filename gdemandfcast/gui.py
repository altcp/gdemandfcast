import os

import streamlit as st

from gdemandfcast.ai import execute


def create():

    st.title("Univariate Forecasting")
    st.subheader("Proof of Concept Demostration")
    st.markdown("***")
    st.markdown("***")
    gear = st.radio("Model Selection::", ("Compare", "Auto"))
    shift = st.selectbox("Type of Models:", ("ML", "TS", "DL"))
    lags = st.slider("Number of Previous Timsteps to Consider:", 1, 5, 3, 1)
    train = st.file_uploader("Upload Training Data: ")
    test = st.file_uploader("Uploat Testing Data: ")
    st.markdown("***")
    gear = gear.lower()
    shift = shift.lower()

    if train is not None and test is not None:

        gear = gear.lower()
        shift = shift.lower()
        df = execute(train, test, lags, gear, shift, "fast", False).frm()

        if shift == "auto":
            a = 1
        else:
            a = 4

        for i in range(len(df)):
            st.line_chart(df.loc[:, i : i + a])
            st.bar_chart(df.loc[:, i : i + a])

    else:

        submit = st.button("Run Demo")
        if submit:
            st.write("Demostration Based on Seen Data.")
            df = execute(train, test, lags, gear, shift, "fast", False).frm()

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
