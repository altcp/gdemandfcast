"""Tests for hello function."""
import pandas as pd

from gdemandfcast import ai


# Get
def get(train, test, lags):
    train1 = pd.read_excel(train)
    test1 = pd.read_excel(test)
    train2 = train1.fillna(0)
    test2 = test1.fillna(0)
    train_X, train_y, test_X, test_y = ai.execute(train2, test2, lags).get()

    return train_X, train_y, test_X, test_y


# Test Manual ML
def test_execute_manualml():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    train_X, train_y, test_X, test_y = get(train, test, 3)
    df, percentage_accurate = ai.automate(
        train_X, train_y, test_X, test_y, "manual", "ml", "fast"
    ).run()
    assert not df.empty


# Test Auto ML
def test_execute_automl():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    train_X, train_y, test_X, test_y = get(train, test, 3)
    df, percentage_accurate = ai.automate(
        train_X, train_y, test_X, test_y, "auto", "ml", "fast"
    ).run()
    assert not df.empty
