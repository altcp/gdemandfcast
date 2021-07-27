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
import sklearn.gaussian_process as gp
import tensorflow as tf
from scipy import stats
from sklearn import model_selection
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR
from tensorflow import keras
from xgboost import XGBRegressor

# %%


class execute:
    def __init__(self, train, test, lags, group="ml", runtype="manual"):
        self.train = train
        self.test = test
        self.lags = lags
        self.group = group
        self.runtype = runtype

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
            T = (
                preprocessing(test2, target, self.lags, True)
                .run_univariate()
                .dropna()
                .reset_index(drop=True)
            )
            y = df2["Y"]
            X = df2.loc[:, df2.columns != "Y"]

            label = target + "_test"
            size = len(T)
            df[label] = y

            if self.runtype == "auto":

                if self.group == "ml":
                    labelpred = target + "_ml"
                    df[labelpred] = compare(X, y, T, False).automl()

                elif self.group == "dl":
                    labelpred = target + "_dl"
                    df[labelpred] = compare(X, y, T, False).autodl()

                else:
                    labelpred = target + "_ts"
                    df[labelpred] = compare(X, y, T, False).autots()

            else:

                if self.group == "ml":
                    labelpred = target + "_ml"
                    pred_df = compare(X, y, T, True).compare_ml()
                    df[labelpred] = pred_df.head(size)
                elif self.group == "dl":
                    labelpred = target + "_dl"
                    pred_df = compare(X, y, T, True).compare_dl()
                    df[labelpred] = pred_df.head(size)

                else:
                    labelpred = target + "_ts"
                    pred_df = compare(X, y, T, True).compare_ts()
                    df[labelpred] = pred_df.head(size)

        return df


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


class compare:
    def __init__(self, X, y, T, charts):
        self.X = X
        self.y = y
        self.T = T
        self.charts = charts

    def compare_ml(self):
        warnings.filterwarnings("ignore")

        m1 = mlmodels(self.X, self.y, False).gpr_model()
        m2 = mlmodels(self.X, self.y, False).mlp_model()
        m3 = mlmodels(self.X, self.y, False).xgb_model()
        m4 = mlmodels(self.X, self.y, False).svr_model()

        df = pd.DataFrame()
        df["Test"] = self.T

        for model, name in (m1, m2, m3, m4):
            df[name] = model.predict(self.T)

        if self.charts == True:
            df.plot.line()

        return df

    def compare_dl(self):

        m1 = dlmodels(1, self.X, self.y, False).run()
        m2 = dlmodels(2, self.X, self.y, False).run()
        m3 = dlmodels(3, self.X, self.y, False).run()
        m4 = dlmodels(4, self.X, self.y, False).run()

        df = pd.DataFrame()
        df["Test"] = self.T

        for model, name in (m1, m2, m3, m4):
            df[name] = model.predict(self.T)

        if self.charts == True:
            df.plot.line()

        return df

    def compare_ts(self):
        # Todo: Rewrite
        pass

    def autots(self):
        # Todo: Rewrite
        pass

    def automl(self):
        # Todo: Rewrite
        pass

    def autodl(self):
        # Todo: Rewrite
        pass


# %%


class mlmodels:
    def __init__(self, X, y, validate, aware="mean", speed="fast"):
        self.x = X
        self.y = y
        self.aware = aware
        self.speed = speed
        self.validate = validate

        self.jobs = -1
        self.seed = 232
        self.scoring = "r2"

    def gpr_model(self):

        gc.collect()
        if self.aware == "mean":
            pipe = Pipeline(
                steps=[
                    ("SCAL", StandardScaler()),
                    ("NORM", MinMaxScaler()),
                    ("GPR", gp.GaussianProcessRegressor()),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("SCAL", RobustScaler()),
                    ("NORM", MinMaxScaler()),
                    ("GPR", gp.GaussianProcessRegressor()),
                ]
            )

        if self.speed == "fast":
            param_grid = {"GPR__alpha": [0.03, 0.05, 0.07]}

        else:
            param_grid = {"GPR__alpha": [0.03, 0.05, 0.07]}

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

        gc.collect()
        if self.aware == "mean":
            pipe = Pipeline(
                steps=[
                    ("SCAL", StandardScaler()),
                    ("NORM", MinMaxScaler()),
                    ("MLP", MLPRegressor()),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("SCAL", RobustScaler()),
                    ("NORM", MinMaxScaler()),
                    ("MLP", MLPRegressor()),
                ]
            )

        if self.speed == "fast":
            param_grid = {
                "MLP__hidden_layer_sizes": [(12, 4), (10,)],
                "MLP__activation": ["tanh", "relu"],
                "MLP__solver": ["sgd", "adam"],
                "MLP__alpha": [0.001, 0.005, 0.01],
                "MLP__learning_rate": ["constant", "adaptive"],
                "MLP__early_stopping": [True],
                "MLP__random_state": [self.seed],
            }

        else:
            param_grid = {
                "MLP__hidden_layer_sizes": [(12, 4), (10,)],
                "MLP__activation": ["tanh", "relu"],
                "MLP__solver": ["sgd", "adam"],
                "MLP__alpha": [0.001, 0.005, 0.01],
                "MLP__learning_rate": ["constant", "adaptive"],
                "MLP__early_stopping": [True],
                "MLP__random_state": [self.seed],
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

        gc.collect()
        if self.aware == "mean":
            pipe = Pipeline(
                steps=[
                    ("SCAL", StandardScaler()),
                    ("NORM", MinMaxScaler()),
                    ("XGB", XGBRegressor(objective="reg:squarederror")),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("SCAL", RobustScaler()),
                    ("NORM", MinMaxScaler()),
                    ("XGB", XGBRegressor(objective="reg:squarederror")),
                ]
            )

        if self.speed == "fast":
            param_grid = {
                "XGB__max_depth": [3, 7],
            }

        else:
            param_grid = {
                "XGB__max_depth": [3, 7],
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

        gc.collect()
        pipe = Pipeline(steps=[("STD", StandardScaler()), ("SVR", SVR())])
        param_grid = {
            "SVR__kernel": ["rbf", "poly"],
            "SVR__C": [1, 50, 100],
            "SVR__epsilon": [0.0001, 0.0005, 0.001],
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


class dlmodels:
    def __init__(self, i, X, y, validation):
        self.i = i
        self.X = X
        self.y = y
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
                shapiro_test = stats.shapiro(data)
                lilliefors_test = stats.diagnostic.lilliefors(data)

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
