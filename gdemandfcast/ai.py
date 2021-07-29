# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import gc
import math
import warnings

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import scipy.stats as sps
import tensorflow as tf
from sklearn import model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.svm import SVR
from statsmodels.stats import diagnostic
from tensorflow import keras
from xgboost import XGBRegressor

# %%


class preprocessing:
    def __init__(self, df, target="Y", p=7, create_testset=False):
        self.df = df
        self.target = target
        self.p = p  # Size of Bucket (e.g., Week = 7 or 5, Lag)
        self.create_testset = create_testset

    def run_univariate(self):

        df1 = pd.DataFrame()

        if self.create_testset == False:
            P = self.p + 1
        else:
            P = self.p

        for i in range(P):
            if i == 0:
                if P > self.p:
                    df1["Y"] = self.df[self.target]
                else:
                    df1["X0"] = self.df[self.target]
            else:
                column_name = "X" + str(i)
                df1[column_name] = self.df[self.target].shift(i)

        return df1

    def create_frequency(self):

        df1 = pd.DataFrame()
        df1 = self.df.groupby(np.arrange(len(self.df)) // self.p).sum()
        df1.index = self.df.loc[1 :: self.p, self.target]

        return df1


class execute:
    def __init__(self, train, test, lags, gear, shift, speed, charts):
        self.train = train
        self.test = test
        self.lags = lags
        self.gear = gear
        self.shift = shift
        self.speed = speed
        self.charts = charts

    def frm(self):

        train1 = pd.read_excel(self.train)
        test1 = pd.read_excel(self.test)
        train2 = train1.fillna(0)
        test2 = test1.fillna(0)

        df = pd.DataFrame()

        for col in train2.columns:

            target = col

            df2 = (
                preprocessing(train2, target, self.lags, False)
                .run_univariate()
                .dropna()
                .reset_index(drop=True)
            )

            test_X = (
                preprocessing(test2, target, self.lags, True)
                .run_univariate()
                .dropna()
                .reset_index(drop=True)
            )

            test_y = test2[target].tail(len(test_X)).reset_index(drop=True)
            train_y = df2["Y"]
            train_X = df2.loc[:, df2.columns != "Y"]

            pred_df, percentage_accurate = automate(
                train_X,
                train_y,
                test_X,
                test_y,
                self.gear,
                self.shift,
                self.speed,
                self.charts,
            ).run()

            if self.gear == "auto":

                for col in pred_df.columns:
                    if col != "Y":
                        n_y = str(target) + "_Y"
                        n_1 = str(target) + "_" + col
                        last_col = col

                df = pd.concat([df, pred_df], axis=1)
                df = df.rename(
                    columns={
                        "Y": n_y,
                        last_col: n_1,
                    }
                )
                print("% Accurate: " + str(percentage_accurate))

            else:

                df = pd.concat([df, pred_df], axis=1)
                for col in pred_df.columns:
                    new_name = str(target) + "_" + col
                    df.rename(columns={col: new_name})

                df = pred_df

        return df


class automate(execute):
    def __init__(self, train_X, train_y, test_X, test_y, gear, shift, speed, charts):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.gear = gear
        self.shift = shift
        self.speed = speed
        self.charts = charts

    def run(self):
        best_mape = 100

        if self.gear == "auto":

            if self.shift == "ml":
                best_model = "XGB"
                df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                    False,
                ).compare_ml()
            elif self.shift == "dl":
                best_model = "GDF-LSTM"
                df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                    False,
                ).compare_dl()
            elif self.shit == "ts":
                best_model = "TS-ES-RNN"
                df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                    False,
                ).compare_ts()
            else:
                best_model = "TS-ES-RNN"
                df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                    False,
                ).compare_auto()

            for col in df.columns:
                if col != "Y":
                    mape = mean_absolute_percentage_error(df["Y"], df[col])
                    if mape < best_mape:
                        best_mape = round(mape, 4)
                        best_model = col

            return_df = df[["Y", best_model]]

        else:

            if self.shift == "ml":
                pred_df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                    self.charts,
                ).compare_ml()

            elif self.shift == "dl":
                pred_df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                    self.charts,
                ).compare_dl()

            elif self.shift == "ts":
                pred_df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                    self.charts,
                ).compare_ts()

            else:
                pred_df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                    self.charts,
                ).compare_auto()

            return_df = pred_df

        # See Magnitude of Absolute Difference
        if self.charts == True:
            print(" ")
            df.plot(figsize=(15, 10), kind="line")
            df.plot(figsize=(15, 10), kind="bar", stacked=False)
            print(
                "Selected"
                + self.shift.upper()
                + "Model: "
                + col
                + " , MAPE: "
                + str(best_mape)
            )
            print(" ")

            if best_mape > 1:
                percentage_accurate = 0
            else:
                percentage_accurate = (1 - best_mape) * 100

        return return_df, percentage_accurate


class compare(automate):
    def __init__(self, train_X, train_y, test_X, test_y, shift, speed, charts):
        super().__init__(train_X, train_y, test_X, test_y, shift, speed, charts)
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.speed = speed
        self.charts = charts

    def compare_ml(self):

        warnings.filterwarnings("ignore")

        m1 = mlmodels(self.train_X, self.train_y, self.speed, False).gpr_model()
        m2 = mlmodels(self.train_X, self.train_y, self.speed, False).mlp_model()
        m3 = mlmodels(self.train_X, self.train_y, self.speed, False).xgb_model()
        m4 = mlmodels(self.train_X, self.train_y, self.speed, False).svr_model()

        column_names = ["Y", "GPR", "MLP", "XGB", "SVR"]
        df = pd.DataFrame(columns=column_names)
        # Remove First Element to Match Prediction
        df["Y"] = self.test_y.loc[1:]

        for model, name in (m1, m2, m3, m4):
            mf = model.predict(self.test_X)
            # Remove Last Element to Match Truth
            df[name] = mf[:-1].tolist()

        # See Magnitude of Absolute Difference
        if self.charts == True:
            print(" ")
            df.plot(figsize=(15, 10), kind="line")
            df.plot(figsize=(15, 10), kind="bar", stacked=False)
            print(" ")

        return df

    def compare_dl(self):

        m1 = dlmodels(1, self.X, self.y, False).run()
        m2 = dlmodels(2, self.X, self.y, False).run()
        m3 = dlmodels(3, self.X, self.y, False).run()
        m4 = dlmodels(4, self.X, self.y, False).run()

        column_names = [
            "Y",
            "GDF-BI_GRU_LTSM",
            "GDF-BI_LSTM",
            "GDF-GRU_LSTM",
            "GDF_LSTM",
        ]
        df = pd.DataFrame(columns=column_names)
        # Remove First Element to Match Prediction
        df["Y"] = self.test_y.loc[1:]

        for model, name in (m1, m2, m3, m4):
            mf = model.predict(self.test_X)
            # Remove Last Element to Match Truth
            df[name] = mf[:-1].tolist()

        # See Magnitude of Absolute Difference
        if self.charts == True:
            print(" ")
            df.plot(figsize=(15, 10), kind="line")
            df.plot(figsize=(15, 10), kind="bar", stacked=False)
            print(" ")

        return df

    def compare_ts(self):
        # Todo: Rewrite
        pass

    def compare_auto(self):
        # Todo: Rewrite
        pass


# %%


class distribution:
    def __init__(self, y):
        self.y = y

    def aware(self):
        data = []
        data = self.y
        shapiro_test = sps.shapiro(data)
        lilliefors_test = diagnostic.lilliefors(data)

        if shapiro_test.pvalue > 0.05:
            if lilliefors_test.pvalue < 0.05:
                distribution = "alt"
            else:
                distribution = "norm"
        else:
            distribution = "alt"

        return distribution


# %%


# %%


class mlmodels(compare):
    def __init__(self, train_X, train_y, speed, validate):
        super().__init__(train_X, train_y, speed, validate)
        self.x = train_X
        self.y = train_y
        self.speed = speed
        self.validate = validate

        self.jobs = -1
        self.scoring = "r2"

    def gpr_model(self):

        gc.collect()
        dist = distribution(self.y).aware()
        if dist == "norm":
            pipe = Pipeline(
                steps=[
                    ("T", PowerTransformer(method="yeo-johnson")),
                    ("N", MinMaxScaler((1, 100))),
                    ("M", GaussianProcessRegressor()),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("S", RobustScaler()),
                    ("N", MinMaxScaler((1, 100))),
                    ("T", PowerTransformer(method="box-cox")),
                    ("M", GaussianProcessRegressor()),
                ]
            )

        if self.speed == "fast":
            param_grid = {
                "M__n_restarts_optimizer": [0, 2, 4, 8],
                "M__alpha": [1e-10, 1e7, 1e-5, 1e-3],
            }
        else:
            ker_rbf = ConstantKernel(1.0, constant_value="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
            ker_rq = ConstantKernel(
                1.0, constant_value_bounds="fixed"
            ) * RationalQuadratic(alph=0.1, length_scale=1)
            ker_ess = ConstantKernel(
                1.0, constant_value_bounds="fixed"
            ) * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
            ker_wk = DotProduct() + WhiteKernel()
            kernel_list = [ker_rbf, ker_rq, ker_ess, ker_wk]

            param_grid = {
                "M__kernel": kernel_list,
                "M__n_restarts_optimizer": [0, 2, 4, 8],
                "M__alpha": [1e-10, 1e7, 1e-5, 1e-3],
            }

        search = GridSearchCV(
            pipe, param_grid, cv=5, scoring=self.scoring, n_jobs=self.jobs
        )
        search.fit(self.x, self.y)
        results = model_selection.cross_val_score(
            search.best_estimator_,
            self.x,
            self.y,
            cv=5,
            scoring="neg_mean_absolute_percentage_error",
        )

        if self.validate == False:
            return search, "GPR"
        else:
            return round((np.nanmean(results) * 100.0), 2)

    def mlp_model(self):

        dist = distribution(self.y).aware()
        if dist == "norm":
            pipe = Pipeline(
                steps=[
                    ("T", PowerTransformer(method="yeo-johnson")),
                    ("N", MinMaxScaler((1, 100))),
                    ("M", MLPRegressor()),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("S", RobustScaler()),
                    ("N", MinMaxScaler((1, 100))),
                    ("T", PowerTransformer(method="box-cox")),
                    ("M", MLPRegressor()),
                ]
            )

        if self.speed == "fast":
            param_grid = {
                "M__hidden_layer_sizes": [(12, 4), (10,)],
                "M__activation": ["relu"],
                "M__solver": ["adam"],
                "M__alpha": [0.001, 0.005, 0.01],
                "M__learning_rate": ["adaptive"],
                "M__early_stopping": [True],
            }
        else:
            param_grid = {
                "M__hidden_layer_sizes": [(3, 3, 3), (12, 4), (10,)],
                "M__activation": ["tanh", "relu"],
                "M__solver": ["sgd", "adam"],
                "M__alpha": [0.001, 0.005, 0.01],
                "M__learning_rate": ["constant", "adaptive"],
                "M__early_stopping": [True],
            }

        search = GridSearchCV(
            pipe, param_grid, cv=5, scoring=self.scoring, n_jobs=self.jobs
        )
        search.fit(self.x, self.y)
        results = model_selection.cross_val_score(
            search.best_estimator_,
            self.x,
            self.y,
            cv=5,
            scoring="neg_mean_absolute_percentage_error",
        )

        if self.validate == False:
            return search, "MLP"
        else:
            return round((np.nanmean(results) * 100.0), 2)

    def xgb_model(self):

        dist = distribution(self.y).aware()
        if dist == "norm":
            pipe = Pipeline(
                steps=[
                    ("T", PowerTransformer(method="yeo-johnson")),
                    ("N", MinMaxScaler((1, 100))),
                    ("M", XGBRegressor(objective="reg:squarederror")),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("S", RobustScaler()),
                    ("N", MinMaxScaler((1, 100))),
                    ("T", PowerTransformer(method="box-cox")),
                    ("M", XGBRegressor(objective="reg:squarederror")),
                ]
            )

        if self.speed == "fast":
            param_grid = {"M__max_depth": [4, 6, 8], "M__eta": [0.05, 0.1, 0.2, 0.3]}
        else:
            param_grid = {
                "M__booster": ["gbtree", "gblinear"],
                "M__max_depth": [4, 6, 8],
                "M__eta": [0.05, 0.1, 0.2, 0.3],
                "M__alpha": [0.1, 0.3, 0.5, 0.7],
                "M__lambda": [1, 1.5, 3.0, 4.5],
            }

        search = GridSearchCV(
            pipe, param_grid, cv=5, scoring=self.scoring, n_jobs=self.jobs
        )
        search.fit(self.x, self.y)
        results = model_selection.cross_val_score(
            search.best_estimator_,
            self.x,
            self.y,
            cv=5,
            scoring="neg_mean_absolute_percentage_error",
        )

        if self.validate == False:
            return search, "XGB"
        else:
            return round((np.nanmean(results) * 100.0), 2)

    def svr_model(self):

        dist = distribution(self.y).aware()
        if dist == "norm":
            pipe = Pipeline(
                steps=[
                    ("T", PowerTransformer(method="yeo-johnson")),
                    ("N", MinMaxScaler((1, 100))),
                    ("M", SVR()),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("S", RobustScaler()),
                    ("N", MinMaxScaler((1, 100))),
                    ("T", PowerTransformer(method="box-cox")),
                    ("M", SVR()),
                ]
            )

        if self.speed == "fast":
            param_grid = {
                "M__C": [1, 3, 5, 7],
                "M__epsilon": [0.001, 0.003, 0.005, 0.01],
            }
        else:
            param_grid = {
                "M__kernel": ["rbf", "poly"],
                "M__Degree": [0, 1, 2, 3],
                "M__C": [1, 3, 5, 7],
                "M__epsilon": [0.001, 0.003, 0.005, 0.01],
            }

        search = GridSearchCV(
            pipe, param_grid, cv=5, scoring=self.scoring, n_jobs=self.jobs
        )
        search.fit(self.x, self.y)
        results = model_selection.cross_val_score(
            search.best_estimator_,
            self.x,
            self.y,
            cv=5,
            scoring="neg_mean_absolute_percentage_error",
        )

        if self.validate == False:
            return search, "SVR"
        else:
            return round((np.nanmean(results) * 100.0), 2)


# %%


class dlmodels(compare):
    def __init__(self, i, train_X, train_y, speed, validate):
        super().__init__(train_X, train_y, speed, validate)
        self.i = i
        self.X = train_X
        self.y = train_y
        self.speed = speed
        self.validation = validate

    def run(self):
        i = self.i
        X = self.X
        y = self.y
        spilt = round((1 / 5), 2)
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=spilt, random_state=232
        )

        def get_tuner(m):

            if m == 1:

                def bi_gru_lstm(hp):
                    model = tf.keras.Sequential()
                    # GRU
                    model.add(
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.GRU(
                                units=hp.Int("neurons_gru", 4, 10, 1, default=7),
                                return_sequences=True,
                            ),
                            input_shape=(test_x.shape[1], 1),
                        )
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    # LSTM
                    model.add(
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.LSTM(
                                units=hp.Int("neurons_lstm", 4, 10, 1, default=7)
                            )
                        )
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    # DENSE
                    model.add(
                        tf.keras.layers.Dense(
                            units=hp.Int("neurons_dense", 4, 10, 1, default=7),
                            activation="relu",
                        )
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.Dense(1))
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(),
                        loss=tf.keras.losses.MSE(),
                        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
                    )
                    return model

                tuner = ModelTuner(
                    oracle=kt.oracles.BayesianOptimization(
                        objective=kt.Objective("loss", "min"), max_trials=3
                    ),
                    hypermodel=bi_gru_lstm,
                    project_name="gdf_bi_gru_ltsm",
                )
                gc.collect()

            elif m == 2:

                def bi_lstm(hp):
                    model = tf.keras.Sequential()
                    # LSTM
                    model.add(
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.LSTM(
                                units=hp.Int("neurons_lstm", 4, 10, 1, default=7),
                                return_sequences=True,
                            ),
                            input_shape=(test_x.shape[1], 1),
                        )
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    # DENSE
                    model.add(
                        tf.keras.layers.Dense(
                            units=hp.Int("neurons_dense", 4, 10, 1, default=7),
                            activation="relu",
                        )
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.Dense(1))
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(),
                        loss=tf.keras.losses.MSE(),
                        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
                    )
                    return model

                tuner = ModelTuner(
                    oracle=kt.oracles.BayesianOptimization(
                        objective=kt.Objective("loss", "min"), max_trials=3
                    ),
                    hypermodel=bi_lstm,
                    project_name="gdf_bi_lstm",
                )
                gc.collect()

            elif m == 3:

                def gru_lstm(hp):
                    model = tf.keras.Sequential()
                    # GRU
                    model.add(
                        tf.keras.layers.GRU(
                            units=hp.Int("neurons_gru", 4, 10, 1, default=7),
                            return_sequences=False,
                        ),
                        input_shape=(test_x.shape[0], 1),
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    # LSTM
                    model.add(
                        tf.keras.layers.LSTM(
                            units=hp.Int("neurons_lstm", 4, 10, 1, default=7)
                        )
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    # DENSE
                    model.add(
                        tf.keras.layers.Dense(
                            units=hp.Int("neurons_dense", 4, 10, 1, default=7),
                            activation="relu",
                        )
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.Dense(1))
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(),
                        loss=tf.keras.losses.MSE(),
                        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
                    )
                    return model

                tuner = ModelTuner(
                    oracle=kt.oracles.BayesianOptimization(
                        objective=kt.Objective("loss", "min"), max_trials=3
                    ),
                    hypermodel=gru_lstm,
                    project_name="gdf_gru_lstm",
                )
                gc.collect()

            else:

                def lstm(hp):
                    model = tf.keras.Sequential()
                    # LSTM
                    model.add(
                        tf.keras.layers.LSTM(
                            units=hp.Int("neurons_lstm", 4, 10, 1, default=7),
                            return_sequences=False,
                        ),
                        input_shape=(test_x.shape[0], 1),
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    # DENSE
                    model.add(
                        tf.keras.layers.Dense(
                            units=hp.Int("neurons_dense", 4, 10, 1, default=7),
                            activation="relu",
                        )
                    )
                    model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.Dense(1))
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(),
                        loss=tf.keras.losses.MSE(),
                        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
                    )
                    return model

                tuner = ModelTuner(
                    oracle=kt.oracles.BayesianOptimization(
                        objective=kt.Objective("loss", "min"), max_trials=3
                    ),
                    hypermodel=lstm,
                    project_name="gdf_lstm",
                )
                gc.collect()

            return tuner

        get_tuner(i).search(X, y)
        best_hps = get_tuner(i).get_best_hyperparameters()[0]
        model = get_tuner(i).hypermodel.build(best_hps)

        call_back = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=3, verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, verbose=0),
        ]

        if (i == 1) or (i == 2):
            X = X.reshape(X.shape[1], 1)
            test_x = test_x.reshape(test_x.shape[1], 1)
        else:
            X = X.reshape(X.shape[0], X.shape[1], 1)
            test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)

        history = model.fit(
            X,
            y,
            validation_data=(test_x, test_y),
            callbacks=call_back,
            epochs=30,
            batch_size=32,
            verbose=0,
            # Set to True if GPU or TPU
            use_multiprocessing=False,
        )
        scores = model.evaluate(test_x, test_y, verbose=0)

        if self.validation == True:

            def get_name(m):

                if m == 1:
                    name = "GDF-BI_GRU_LTSM"
                elif m == 2:
                    name = "GDF-BI_LSTM"
                elif m == 3:
                    name = "GDF-GRU_LSTM"
                else:
                    name = "GDF_LSTM"

                return name

            visualization(
                history, round((scores[1] * 100), 2), get_name(self.i)
            ).disp_fit()
            print(" ")
            print(" ")
            return None

        else:

            return model


# %%


class visualization:
    def __init__(self, history, score, model):
        self.history = history
        self.score = score
        self.model = model

    def disp_fit(self):

        print(" ")
        print(" ")
        msg = f"{self.model} achieved a MAPE of {self.score} using Distribution Aware Gradient Descent Optimization."
        print(msg)

        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.legend(["training loss", "validation loss"])
        plt.show()

        return None


# %%


class ModelTuner(kt.Tuner):
    def run_trial(self, trial, x_train, y_train):
        batch_size = 32
        hp = trial.hyperparameters
        model = self.hypermodel.build(trial.hyperparameters)
        epoch_loss_metric = tf.keras.metrics.Mean()
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Float(
                "opt_learn_rate",
                min_value=1e-4,
                max_value=1e-2,
                sampling="LOG",
                default=1e-3,
            ),
            clipnorm=hp.Float(
                "opt_clipnorm", min_value=0.001, max_value=1.11, step=0.10, default=1.0
            ),
            clipvalue=hp.Float(
                "opt_clipvalue", min_value=1, max_value=5.50, step=0.25, default=5.0
            ),
        )

        @tf.function
        def run_train_step(real_x, real_y):

            with tf.GradientTape() as tape:

                pred_y = model(real_x)

                data = []
                data = real_y - pred_y
                shapiro_test = sps.shapiro(data)
                lilliefors_test = diagnostic.lilliefors(data)

                dev = []
                dev = abs(real_y - pred_y)
                q3, q1 = np.percentile(dev, [75, 25])
                iqr = q3 - q1
                d = q3 + (1.5 * iqr)

                huber = tf.keras.losses.Huber(delta=d)
                mse = tf.keras.losses.MSE()

                # Distribution Aware
                if shapiro_test.pvalue > 0.05:
                    if lilliefors_test.pvalue < 0.05:
                        loss = huber(real_y, pred_y)
                    else:
                        loss = mse(real_y, pred_y)
                else:
                    loss = huber(real_y, pred_y)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)

            return loss

        # Calculate number of batches and define number of epochs per Trial
        num_of_batches = math.floor(len(x_train) / batch_size)
        epochs = 10

        # Run the Trial
        for epoch in range(epochs):
            self.on_epoch_begin(trial, model, epoch, logs={})
            for batch in range(num_of_batches):
                n = batch * batch_size
                self.on_batch_begin(trial, model, batch, logs={})
                batch_loss = run_train_step(
                    x_train[n : n + batch_size], y_train[n : n + batch_size]
                )
                print(float(batch_loss))
                self.on_batch_end(trial, model, batch, logs={"loss": float(batch_loss)})

            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epoch, logs={"loss": epoch_loss})
            epoch_loss_metric.reset_states()


# %%
