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
from keras_tuner import tuners
from numpy.core.numeric import False_
from pmdarima.arima.auto import AutoARIMA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
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


class execute:
    def __init__(self, train, test, lags):
        self.train = train
        self.test = test
        self.lags = lags

    def get(self):

        target = self.train.columns[0]

        df2 = pd.DataFrame()
        df2 = (
            preprocessing(self.train, target, self.lags, False)
            .run_univariate()
            .dropna()
            .reset_index(drop=True)
        )
        test_X = (
            preprocessing(self.test, target, self.lags, True)
            .run_univariate()
            .dropna()
            .reset_index(drop=True)
        )

        test_y = self.test[target].tail(len(test_X)).reset_index(drop=True)
        train_y = df2["Y"]
        train_X = df2.loc[:, df2.columns != "Y"]

        return train_X, train_y, test_X, test_y

    def rescale(self):
        def create_dataset(dataset, look_back=1):

            dataset = np.asarray(dataset)
            dataX, dataY = list(), list()

            for i in range(len(dataset) - look_back):
                a = dataset[i : (i + look_back)]
                dataX.append(a)
                dataY.append(dataset[i + look_back])

            return np.asarray(dataX), np.asarray(dataY)

        train_X, train_y = create_dataset(self.train, self.lags)
        test_X, test_y = create_dataset(self.test, self.lags)

        return train_X, train_y, test_X, test_y


# %%


class automate:
    def __init__(self, train_X, train_y, test_X, test_y, gear, shift, speed):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.gear = gear
        self.shift = shift
        self.speed = speed

    def run(self):

        if self.gear == "auto":

            best_rmse = 100000
            df = pd.DataFrame()

            if self.shift == "ml":
                best_model = "GPR"
                df = compare(
                    self.train_X, self.train_y, self.test_X, self.test_y, self.speed
                ).compare_ml()

            else:
                best_model = "GRU"
                df = compare(
                    self.train_X, self.train_y, self.test_X, self.test_y, self.speed
                ).compare_dl()

            for col in df.columns:

                if col != "Y":
                    mse = mean_squared_error(df["Y"], df[col])
                    rmse = math.sqrt(mse)
                    if rmse < best_rmse:
                        best_rmse = round(rmse, 4)
                        best_model = col

            df1 = df[["Y", best_model]].reset_index(drop=True)
            df2 = df1.round(4)
            return_df = df2

        else:

            if self.shift == "ml":
                pred_df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                ).compare_ml()

            else:
                pred_df = compare(
                    self.train_X,
                    self.train_y,
                    self.test_X,
                    self.test_y,
                    self.speed,
                ).compare_dl()

            df1 = pred_df
            df2 = df1.round(4)
            return_df = df2

        return return_df


class compare:
    def __init__(self, train_X, train_y, test_X, test_y, speed):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.speed = speed

    def compare_ml(self):

        warnings.filterwarnings("ignore")

        m1 = mlmodels(self.train_X, self.train_y, self.speed).gpr_model()
        m2 = mlmodels(self.train_X, self.train_y, self.speed).knn_model()
        m3 = mlmodels(self.train_X, self.train_y, self.speed).xgb_model()
        m4 = mlmodels(self.train_X, self.train_y, self.speed).svr_model()

        column_names = ["Y", "GPR", "KNN", "XGB", "SVR"]
        df = pd.DataFrame(columns=column_names)
        # Remove First Element to Match Prediction
        df["Y"] = self.test_y.loc[1:]

        for model, name in (m1, m2, m3, m4):
            mf = model.predict(self.test_X)
            # Remove Last Element to Match Prediction
            df[name] = mf[:-1].tolist()

        return df

    def compare_dl(self):

        m1 = dlmodels(1, self.train_X, self.train_y, self.speed).run()
        m2 = dlmodels(2, self.train_X, self.train_y, self.speed).run()
        m3 = dlmodels(3, self.train_X, self.train_y, self.speed).run()
        m4 = dlmodels(4, self.train_X, self.train_y, self.speed).run()

        column_names = [
            "Y",
            "BI_GRU_GRU",
            "GRU_GRU",
            "BI_GRU",
            "GRU",
        ]

        df = pd.DataFrame(columns=column_names)
        # Remove First Element to Match Prediction
        yf = self.test_y[1:]
        yf2 = yf.ravel()
        df["Y"] = yf2.tolist()

        for model, name in (m1, m2, m3, m4):
            # Remove Last Element to Match Prediction
            mf = model.predict(self.test_X, verbose=0)
            mf2 = mf.ravel()
            df[name] = mf2[:-1].tolist()

        return df


class regress:
    def __init__(self, train_df, test_df, horizon=1, seed=232):
        self.train_df = train_df
        self.test_df = test_df
        self.horizon = horizon
        self.seed = seed

    def manual_sm(self):

        column_names = ["Y", "ARMA", "ARIMA", "SARIMA"]
        df = pd.DataFrame(columns=column_names)
        df["Y"] = self.test_df

        mt = self.train_df
        for i in range(len(self.test_df)):

            p1 = smmodels(mt, False, self.horizon, self.seed).arma()
            p2 = smmodels(mt, False, self.horizon, self.seed).arima()
            p3 = smmodels(mt, True, self.horizon, self.seed).arima()
            p4 = smmodels(mt, False, self.horizon, self.seed).frima()

            df.at[i, "ARMA"] = p1
            df.at[i, "ARIMA"] = p2
            df.at[i, "SARIMA"] = p3
            df.at[i, "FRIMA"] = p4

            sample = df.iloc[i]["Y"]

            pos = len(mt) + i
            mt.at[pos, mt.columns[0]] = sample
            # print(mt)

        # print(df)
        df["Y"] = df["Y"].shift(-1)
        df1 = df.dropna().reset_index(drop=True)
        df2 = df1.round(4)

        return df2

    def auto_sm(self):

        best_rmse = 100000
        best_model = "SARIMA"

        df = self.manual_sm()
        for col in df.columns:

            if col != "Y":
                mse = mean_squared_error(df["Y"], df[col])
                rmse = math.sqrt(mse)
                if rmse < best_rmse:
                    best_rmse = round(rmse, 4)
                    best_model = col

        df1 = df[["Y", best_model]].reset_index(drop=True)
        df2 = df1.round(4)

        return df2


# %%
class smmodels:
    def __init__(self, df, seasonal, horizon, seed, periodicity=12, alpha=0.05):
        self.y = df
        self.seasonal = seasonal
        self.horizon = horizon
        self.seed = seed
        self.periodicity = periodicity
        self.alpha = alpha

    def arima(self):

        kpss_test = pm.arima.ndiffs(self.y, alpha=self.alpha, test="kpss", max_d=4)
        adf_test = pm.arima.ndiffs(self.y, alpha=self.alpha, test="adf", max_d=4)
        num_of_diffs = max(kpss_test, adf_test)

        search = pm.auto_arima(
            self.y,
            d=num_of_diffs,
            start_p=0,
            start_q=0,
            start_P=0,
            max_p=4,
            max_q=4,
            trace=False,
            seasonal=self.seasonal,
            error_action="ignore",
            random_state=self.seed,
            suppress_warnings=True,
        )

        try:

            if self.horizon == 1:
                e_mu = search.predict(n_periods=1)
                e_mu = e_mu[0]

                return e_mu

            else:
                df = pd.DataFrame()
                for i in range(1, (self.horizon + 1), 1):
                    e_mu = search.predict(n_periods=i)
                    e_mu = e_mu[0]
                    df.append(e_mu)

                return df

        except:

            e_mu = 0
            return e_mu

    def frima(self):

        pipe = Pipeline(
            [
                (
                    "fourier",
                    pm.preprocessing.FourierFeaturizer(m=self.periodicity, k=4),
                ),
                (
                    "arima",
                    AutoARIMA(
                        stepwise=True,
                        trace=1,
                        error_action="ignore",
                        seasonal=self.seasonal,
                        suppress_warnings=True,
                    ),
                ),
            ]
        )

        try:

            if self.horizon == 1:
                e_mu = pipe.predict(n_periods=1)
                e_mu = e_mu[0]

                return e_mu

            else:
                df = pd.DataFrame()
                for i in range(1, (self.horizon + 1), 1):
                    e_mu = pipe.predict(n_periods=i)
                    e_mu = e_mu[0]
                    df.append(e_mu)

                return df

        except:

            e_mu = 0
            return e_mu

    def arma(self):

        search = pm.auto_arima(
            self.y,
            d=0,
            start_p=0,
            start_q=0,
            start_P=0,
            max_p=4,
            max_q=4,
            trace=False,
            seasonal=self.seasonal,
            error_action="ignore",
            random_state=self.seed,
            suppress_warnings=True,
        )

        try:

            if self.horizon == 1:
                e_mu = search.predict(n_periods=1)
                e_mu = e_mu[0]

                return e_mu

            else:
                df = pd.DataFrame()
                for i in range(1, (self.horizon + 1), 1):
                    e_mu = search.predict(n_periods=i)
                    e_mu = e_mu[0]
                    df.append(e_mu)

                return df

        except:

            e_mu = 0
            return e_mu


class mlmodels:
    def __init__(self, train_X, train_y, speed):
        self.x = train_X
        self.y = train_y
        self.speed = speed

        self.jobs = -1
        self.scoring = "r2"

    def gpr_model(self):

        gc.collect()
        dist = distribution(self.y).aware()
        if dist == "norm":
            pipe = Pipeline(
                steps=[
                    ("T", PowerTransformer(method="yeo-johnson")),
                    ("M", GaussianProcessRegressor()),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("S", RobustScaler()),
                    ("T", PowerTransformer(method="yeo-johnson")),
                    ("M", GaussianProcessRegressor()),
                ]
            )

        if self.speed == "fast":
            param_grid = {
                "M__n_restarts_optimizer": [2, 4, 8],
                "M__random_state": [232],
            }

        else:
            ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
            ker_rq = ConstantKernel(
                1.0, constant_value_bounds="fixed"
            ) * RationalQuadratic(alpha=0.1, length_scale=1)
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
        return search, "GPR"

    def knn_model(self):

        pipe = Pipeline(
            steps=[
                ("N", MinMaxScaler((1, 100))),
                ("M", KNeighborsRegressor()),
            ]
        )

        if self.speed == "fast":
            param_grid = {"M__n_neighbors": [3, 6, 9]}
        else:
            param_grid = {"M__n_neighbors": [1, 3, 5, 7, 9]}

        search = GridSearchCV(
            pipe, param_grid, cv=5, scoring=self.scoring, n_jobs=self.jobs
        )
        search.fit(self.x, self.y)
        return search, "KNN"

    def xgb_model(self):

        pipe = Pipeline(
            steps=[
                ("N", MinMaxScaler((1, 100))),
                ("M", XGBRegressor(objective="reg:squarederror")),
            ]
        )

        if self.speed == "fast":
            param_grid = {"M__eta": [0.05, 0.1, 0.2, 0.3]}
        else:
            param_grid = {
                "M__booster": ["gbtree", "gblinear"],
                "M__eta": [0.05, 0.1, 0.2, 0.3],
                "M__alpha": [0.1, 0.3, 0.5, 0.7],
                "M__lambda": [1, 1.5, 3.0, 4.5],
            }

        search = GridSearchCV(
            pipe, param_grid, cv=5, scoring=self.scoring, n_jobs=self.jobs
        )
        search.fit(self.x, self.y)
        return search, "XGB"

    def svr_model(self):

        pipe = Pipeline(
            steps=[
                ("N", MinMaxScaler((1, 100))),
                ("M", SVR()),
            ]
        )

        if self.speed == "fast":
            param_grid = {"M__epsilon": [0.001, 0.003, 0.005, 0.01]}
        else:
            param_grid = {
                "M__kernel": ["rbf", "poly"],
                "M__C": [0.5, 1.0, 1.5, 2.0],
                "M__degree": [0, 1, 2, 3],
                "M__epsilon": [0.001, 0.003, 0.005, 0.007, 0.01],
            }

        search = GridSearchCV(
            pipe, param_grid, cv=5, scoring=self.scoring, n_jobs=self.jobs
        )
        search.fit(self.x, self.y)
        return search, "SVR"


# %%
class dlmodels:
    def __init__(self, i, train_X, train_y, speed="fast", lags=3, validation=False):
        self.i = i
        self.X = train_X
        self.y = train_y
        self.speed = speed
        self.lags = lags
        self.validation = validation

    def run(self):

        # Callbacks
        call_back = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=3, verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, verbose=0),
        ]

        # Splits
        train_X, test_X, train_y, test_y = train_test_split(
            self.X, self.y, test_size=0.3
        )

        def get_model(m):

            if m == 1:

                def bi_gru_gru(hp):
                    model = tf.keras.Sequential()

                    # GRU
                    model.add(
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.GRU(
                                units=hp.Int("neurons_gru", 4, 10, 1, default=7),
                                return_sequences=True,
                            ),
                            input_shape=(self.lags, 1),
                        )
                    )
                    model.add(tf.keras.layers.BatchNormalization())

                    # GRU
                    model.add(
                        tf.keras.layers.GRU(
                            units=hp.Int("neurons_gru", 4, 10, 1, default=7),
                        ),
                    )
                    model.add(tf.keras.layers.BatchNormalization())

                    model.add(tf.keras.layers.Dense(1))
                    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
                    return model

                if self.speed == "fast":

                    tuner = kt.Hyperband(
                        bi_gru_gru,
                        objective="mse",
                        executions_per_trial=3,
                        max_epochs=10,
                        project_name="gdf_bi_gru_gru_fast",
                    )
                    tuner.search(self.X, self.y)
                    tuned_model = tuner.get_best_models()[0]

                else:

                    tuner = ModelTuner(
                        oracle=kt.oracles.BayesianOptimization(
                            objective=kt.Objective("loss", "min"), max_trials=3
                        ),
                        hypermodel=bi_gru_gru,
                        project_name="gdf_bi_gru_gru_slow",
                    )
                    tuner.search(self.X, self.y)
                    tuned_model = tuner.get_best_models()[0]

            elif m == 2:

                def gru_gru(hp):
                    model = tf.keras.Sequential()

                    # GRU
                    model.add(
                        tf.keras.layers.GRU(
                            units=hp.Int("neurons_gru", 4, 10, 1, default=7),
                            input_shape=(self.lags, 1),
                            return_sequences=True,
                        ),
                    )
                    model.add(tf.keras.layers.BatchNormalization())

                    # GRU
                    model.add(
                        tf.keras.layers.GRU(
                            units=hp.Int("neurons_gru", 4, 10, 1, default=7),
                        ),
                    )
                    model.add(tf.keras.layers.BatchNormalization())

                    model.add(tf.keras.layers.Dense(1))
                    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
                    return model

                if self.speed == "fast":

                    tuner = kt.Hyperband(
                        gru_gru,
                        objective="mse",
                        executions_per_trial=3,
                        max_epochs=10,
                        project_name="gdf_gru_gru_fast",
                    )
                    tuner.search(self.X, self.y)
                    tuned_model = tuner.get_best_models()[0]

                else:

                    tuner = ModelTuner(
                        oracle=kt.oracles.BayesianOptimization(
                            objective=kt.Objective("loss", "min"), max_trials=3
                        ),
                        hypermodel=gru_gru,
                        project_name="gdf_gru_gru_slow",
                    )
                    tuner.search(self.X, self.y)
                    tuned_model = tuner.get_best_models()[0]

            elif m == 3:

                def bi_gru(hp):
                    model = tf.keras.Sequential()
                    # GRU
                    model.add(
                        tf.keras.layers.Bidirectional(
                            tf.keras.layers.GRU(
                                units=hp.Int("neurons_gru", 4, 10, 1, default=7),
                            ),
                            input_shape=(self.lags, 1),
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
                    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
                    return model

                if self.speed == "fast":

                    tuner = kt.Hyperband(
                        bi_gru,
                        objective="mse",
                        executions_per_trial=3,
                        max_epochs=10,
                        project_name="gdf_bi_gru_fast",
                    )
                    tuner.search(self.X, self.y)
                    tuned_model = tuner.get_best_models()[0]

                else:

                    tuner = ModelTuner(
                        oracle=kt.oracles.BayesianOptimization(
                            objective=kt.Objective("loss", "min"), max_trials=3
                        ),
                        hypermodel=bi_gru,
                        project_name="gdf_bi_gru_slow",
                    )
                    tuner.search(self.X, self.y)
                    tuned_model = tuner.get_best_models()[0]

            else:

                def gru(hp):
                    model = tf.keras.Sequential()
                    # GRU
                    model.add(
                        tf.keras.layers.GRU(
                            units=hp.Int("neurons_gru", 4, 10, 1, default=7),
                            input_shape=(self.lags, 1),
                        ),
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
                    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
                    return model

                if self.speed == "fast":

                    tuner = kt.Hyperband(
                        gru,
                        objective="mse",
                        executions_per_trial=3,
                        max_epochs=10,
                        project_name="gdf_gru_fast",
                    )
                    tuner.search(self.X, self.y)
                    tuned_model = tuner.get_best_models()[0]

                else:

                    tuner = ModelTuner(
                        oracle=kt.oracles.BayesianOptimization(
                            objective=kt.Objective("loss", "min"), max_trials=3
                        ),
                        hypermodel=gru,
                        project_name="gdf_gru_slow",
                    )
                    tuner.search(self.X, self.y)
                    tuned_model = tuner.get_best_models()[0]

            return tuned_model

        # Print Fit
        if self.validation == True:

            model = get_model(self.i)

            history = model.fit(
                train_X,
                train_y,
                validation_data=(test_X, test_y),
                callbacks=call_back,
                verbose=0,
                use_multiprocessing=False,
            )

            def get_name(m):

                if m == 1:
                    name = "BI_GRU_GRU"
                elif m == 2:
                    name = "GRU_GRU"
                elif m == 3:
                    name = "BI_GRU"
                else:
                    name = "GRU"

                return name

            visualization(history).disp_fit()
            print(" ")
            print(" ")

            return None

        else:

            model = get_model(self.i)
            model.fit(
                train_X,
                train_y,
                callbacks=call_back,
                verbose=0,
                use_multiprocessing=False,
            )

            def get_name(m):

                if m == 1:
                    name = "BI_GRU_GRU"
                elif m == 2:
                    name = "GRU_GRU"
                elif m == 3:
                    name = "BI_GRU"
                else:
                    name = "GRU"

                return name

            return (model, get_name(self.i))


# %%
class visualization:
    def __init__(self, history):
        self.history = history

    def disp_fit(self):

        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.legend(["training loss", "validation loss"])
        plt.show()

        return None


# %%
class ModelTuner(kt.Tuner):
    def run_trial(self, trial, x_train, y_train):

        hp = trial.hyperparameters
        model = self.hypermodel.build(trial.hyperparameters)
        epoch_loss_metric = tf.keras.metrics.Mean()
        optimizer = tf.keras.optimizers.Adam(
            lr=hp.Float(
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

                real_y = tf.cast(real_y, dtype="float32")
                pred_y = tf.cast(pred_y, dtype="float32")

                data = []
                data = real_y - pred_y

                dev = []
                dev = abs(data)
                q3, q1 = np.percentile(dev, [75, 25])
                iqr = q3 - q1
                d = q3 + (1.5 * iqr)

                # Distribution Aware
                dist = distribution(data).aware()
                if dist == "norm":
                    mse = tf.keras.losses.MeanAbsoluteError()
                    loss = mse(real_y, pred_y).numpy()
                else:
                    huber = tf.keras.losses.Huber(delta=d)
                    loss = huber(real_y, pred_y).numpy()

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)

            return loss

        # Calculate number of batches and define number of epochs per Trial
        batch_size = 4
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
                self.on_batch_end(trial, model, batch, logs={"loss": batch_loss})

            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epoch, logs={"loss": epoch_loss})
            epoch_loss_metric.reset_states()


# %%
