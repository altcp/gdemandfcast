"""Tests for hello function."""
import pandas as pd

from gdemandfcast.ai import automate, execute


# Get
def get(train, test, lags):
    train1 = pd.read_excel(train)
    test1 = pd.read_excel(test)
    train2 = train1.fillna(0)
    test2 = test1.fillna(0)
    train_X, train_y, test_X, test_y = execute(train2, test2, lags).get()

    return train_X, train_y, test_X, test_y


# Test Manual ML
def test_execute_manualml():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # st.write(train)

        train_X, train_y, test_X, test_y = execute(train, test, 3).get()
        df = automate(train_X, train_y, test_X, test_y, "manual", "ml", "fast").run()

    assert not df.empty


# Test Auto ML
def test_execute_automl():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # st.write(train)

        train_X, train_y, test_X, test_y = execute(train, test, 3).get()
        df = automate(train_X, train_y, test_X, test_y, "auto", "ml", "fast").run()

    assert not df.empty
