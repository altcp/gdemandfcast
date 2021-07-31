import streamlit as st
from ai import execute

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

    df = execute(train, test, lags, gear, shift, speed).frm()

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
        df = execute(train, test, lags, gear, shift, speed).frm()
        st.write("Demostration Based on Seen Data.")
        print(df)

        if gear == "auto":
            a = 1
        else:
            a = 4

        for i in range(len(df)):
            st.line_chart(df.loc[:, i : i + a])
            st.bar_chart(df.loc[:, i : i + a])
