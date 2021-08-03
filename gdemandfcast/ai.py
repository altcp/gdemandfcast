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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.svm import SVR
from statsmodels.stats import diagnostic
from statsmodels.tsa.arima_model import ARMA
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


# %%
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

            best_mape = 100
            df = pd.DataFrame()

            if self.shift == "ml":
                best_model = "GPR"
                df = compare(
                    self.train_X, self.train_y, self.test_X, self.test_y, self.speed
                ).compare_ml()

            elif self.shift == "dl":
                best_model = "GDF-LSTM"
                df = compare(
                    self.train_X, self.train_y, self.test_X, self.test_y, self.speed
                ).compare_dl()

            elif self.shit == "ts":
                best_model = "TS-ES-RNN"
                df = compare(
                    self.train_X, self.train_y, self.test_X, self.test_y, self.speed
                ).compare_ts()

            else:
                best_model = "TS-ES-RNN"
                df = compare(
                    self.train_X, self.train_y, self.test_X, self.test_y, self.speed
                ).compare_auto()

            for col in df.columns:

                if col != "Y":
                    mape = mean_absolute_percentage_error(df["Y"], df[col])
                    if mape < best_mape:
                        best_mape = round(mape, 4)
                        best_model = col

            return_df = df[["Y", best_model]].reset_index(drop=True)

        else:

            if self.shift == "ml":
                pred_df = compare(
                    self.train_X, self.train_y, self.test_X, self.test_y, self.speed
                ).compare_ml()

            elif self.shift == "dl":
                pred_df = compare(
                    self.train_X, self.train_y, self.test_X, self.test_y, self.speed
                ).compare_dl()

            elif self.shift == "ts":
                pred_df = compare(
                    self.train_X, self.train_y, self.test_X, self.test_y, self.speed
                ).compare_ts()

            else:
                pred_df = "Access to Cloud Models Not Ready Yet"

            return_df = pred_df

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
            # Remove Last Element to Match Truth
            df[name] = mf[:-1].tolist()

        return df

    def compare_dl(self):

        m1 = dlmodels(1, self.train_X, self.train_y).run()
        m2 = dlmodels(
            2,
            self.train_X,
            self.train_y,
        ).run()
        m3 = dlmodels(3, self.train_X, self.train_y).run()
        m4 = dlmodels(4, self.train_X, self.train_y).run()

        column_names = [
            "Y",
            "BI_GRU_LTSM",
            "BI_LSTM",
            "GRU_LSTM",
            "GDF_LSTM",
        ]
        df = pd.DataFrame(columns=column_names)
        # Remove First Element to Match Prediction
        df["Y"] = self.test_y.loc[1:]

        for model, name in (m1, m2, m3, m4):
            mf = model.predict(self.test_X)
            # Remove Last Element to Match Truth
            df[name] = mf[:-1].tolist()

        return df


class regress:
    def __init__(
        self,
        train_df,
        test_df,
    ):
        self.train_df = train_df
        self.test_df = test_df

    def manual_sm(self):

        m1 = smmodels(self.train_df, False, 123).arma()
        m2 = smmodels(self.train_df, False, 123).arima()
        m3 = smmodels(self.train_df, True, 123).arima()

        column_names = ["Y", "ARMA", "ARIMA", "SARIMA"]
        df = pd.DataFrame(columns=column_names)
        # Remove First Element to Match Prediction
        df["Y"] = self.test_df.iloc[1:].reset_index(drop=True)

        for model, name in (m1, m2, m3):
            mf = model.predict(self.test_df.shape[0])
            # Remove Last Element to Match Truth
            df[name] = mf[:-1].tolist()

        return df

    def auto_sm(self):

        df = self.manual_sm()
        for col in df.columns:

            if col != "Y":
                mape = mean_absolute_percentage_error(df["Y"], df[col])
                if mape < best_mape:
                    best_mape = round(mape, 4)
                    best_model = col

        return_df = df[["Y", best_model]].reset_index(drop=True)

        return return_df


# %%
class smmodels:
    def __init__(self, df, seasonal, seed, alpha=0.05):
        self.y = df
        self.seasonal = seasonal
        self.seed = seed
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
            max_P=4,
            max_q=4,
            trace=False,
            seasonal=self.seasonal,
            error_action="ignore",
            random_state=self.seed,
            suppress_warnings=True,
        )

        if self.seasonal == True:
            return search, "SARIMA"
        else:
            return search, "ARIMA"

    def arma(self):

        model = ARMA(self.y, order=(1, 1))
        search = model.fit(trend="nc", method="css-mle")

        return search, "ARMA"


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
            param_grid = {"M__n_restarts_optimizer": [2, 4, 8]}

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

        dist = distribution(self.y).aware()
        if dist == "norm":
            pipe = Pipeline(
                steps=[
                    ("T", PowerTransformer(method="yeo-johnson")),
                    ("N", MinMaxScaler((1, 100))),
                    ("M", KNeighborsRegressor()),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("S", RobustScaler()),
                    ("N", MinMaxScaler((1, 100))),
                    ("T", PowerTransformer(method="box-cox")),
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
        return search, "SVR"


# %%
class damodels:
    def __init__(self, i, train_X, train_y, speed, validation):
        super().__init__(train_X, train_y, speed)
        self.i = i
        self.X = train_X
        self.y = train_y
        self.speed = speed
        self.validation = validation

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
