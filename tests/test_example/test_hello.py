""" Automated Tests and (or) Usage Examples """
import pandas as pd

from gdemandfcast.ai import automate, dlmodels, execute, forecast, regress, smmodels


# Get
def get(train, test, lags):
    train1 = pd.read_excel(train)
    test1 = pd.read_excel(test)
    train2 = train1.fillna(0)
    test2 = test1.fillna(0)
    train_X, train_y, test_X, test_y = execute(train2, test2, lags).get()

    return train_X, train_y, test_X, test_y


# Test Manual SM
def test_execute_manualsm():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # print(train)
        df = regress(train, test).manual_sm()

    assert not df.empty


# Test Auto SM
def test_execute_autosm():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # print(train)
        df = regress(train, test).auto_sm()

    assert not df.empty


# Test Forecast SM
def test_execute_one():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)

    data = pd.concat([df_train, df_test], ignore_index=True)

    for col in data.columns:
        df_train = data[[col]]

        mt = df_train
        # MultiStep (e.g., 8), Single Step == 1
        forecast = smmodels(mt, False, 8, 232).arima()

    assert not forecast.empty


# Test Manual ML
def test_execute_manualml():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)
    lags = 3

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # print(train)

        train_X, train_y, test_X, test_y = execute(train, test, lags).get()
        df = automate(train_X, train_y, test_X, test_y, "manual", "ml", lags).run()

    assert not df.empty


# Test Auto ML
def test_execute_automl():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)
    lags = 3

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # print(train)

        train_X, train_y, test_X, test_y = execute(train, test, lags).get()
        df = automate(train_X, train_y, test_X, test_y, "auto", "ml", lags).run()

    assert not df.empty


# Test Forecast ML
def test_execute_one_ml():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)
    lags = 3
    horizon = 8
    forecast_df = pd.DataFrame()

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        forecast_df = forecast(
            forecast_df, train, test, col, lags, horizon
        ).forecast_ml()

    assert not forecast_df.empty


# Test Manual DL Custom
def test_execute_manualdl():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)
    lags = 3

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # print(train)

        train_X, train_y, test_X, test_y = execute(train, test, lags).rescale()
        df = automate(train_X, train_y, test_X, test_y, "manual", "dl", lags).run()

    assert not df.empty


# Test Auto DL Custom
def test_execute_autodl():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)
    lags = 3

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # print(train)

        train_X, train_y, test_X, test_y = execute(train, test, lags).rescale()
        df = automate(train_X, train_y, test_X, test_y, "auto", "dl", lags).run()

    assert not df.empty


# Test Forecast DL
def test_execute_one_dl():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)
    lags = 3
    horizon = 8
    forecast_df = pd.DataFrame()

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        forecast_df = forecast(
            forecast_df, train, test, col, lags, horizon
        ).forecast_dl()

    assert not forecast_df.empty


# Validate DL Model Performance
def test_validate_one():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)
    lags = 3

    for col in df_train.columns:

        train = df_train[[col]].reset_index(drop=True)
        test = df_test[[col]].reset_index(drop=True)
        # print(train)

        train_X, train_y, test_X, test_y = execute(train, test, lags).rescale()
        dlmodels(1, train_X, train_y, lags, True).run()
