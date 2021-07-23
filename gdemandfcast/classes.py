#!/usr/bin/env python
# coding: utf-8

import gc
import math
import arch as am

import pandas as pd
import pmdarima as pm
import keras_tuner as kt
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import sklearn.gaussian_process as gp

from tensorflow import keras
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from scipy import stats



class archmodels:
    
    def __init__(self, y, vol='Garch', dist='Normal', mu='Constant', p=1, o=0, q=1, power=2.0, n=20):
        self.vol = vol
        self.p = p
        self.q = q
        self.o = o
        self.power = power
        self.mu = mu
        self.dist = dist
        self.n = n
        self.y = y
        
        
    def run_model(self):
        
        if (len(self.y) > self.n):
            model = am.arch_model(y=self.y, mean=self.mu, vol=self.vol, dist=self.dist, p=self.p, o=self.o, q=self.q, power=self.power, rescale=True, reindex=False)
            fitted = model.fit(disp='off', show_warning=False)
            summary = fitted.forecast(horizon=1)
            e_sigma_squared = summary.variance.iloc[-1].values
            e_sigma_squared = e_sigma_squared[0]
                
        else: 
                
            print("ERR. Not Enough Data.")
                    
        return e_sigma_squared

# In[ ]:





# In[ ]:

class armamodels:
    
    def __init__(self, y, D=8, P=8, Q=8, seed=232, seasonal=True, alpha=0.05):
        self.y = y
        self.D = D
        self.P = P
        self.Q = Q
        self.seasonal = seasonal
        self.seed = seed
        self.alpha = alpha
    
    
    def run_model(self):
        
        kpss_test = pm.arima.ndiffs(self.y, alpha=self.alpha, test='kpss', max_d=self.D)
        adf_test = pm.arima.ndiffs(self.y, alpha=self.alpha, test='adf', max_d=self.D)
        num_of_diffs = max(kpss_test, adf_test)
    
        arima_model = pm.auto_arima(self.y, d=num_of_diffs, start_p=0, start_q=0, start_P=0, max_p=self.P, max_q=self.Q, trace=False, 
                                    seasonal=self.seasonal, error_action='ignore', random_state=self.seed, suppress_warnings=True)
        
        e_mu = arima_model.predict(n_periods=1)
        e_mu = e_mu[0]
        
        return e_mu

# In[ ]:





# In[ ]:

class mlmodels:
    
    def __init__(self, X, y, cv, validate, scoring='r2', num_of_cpu=-1, seed=232):
        self.x = X
        self.y = y
        self.cv = cv
        self.validate = validate

        self.scoring = scoring
        self.jobs = num_of_cpu
        self.seed = seed

        
    def gpr_model(self):
        
        gc.collect() 
        pipe = Pipeline(steps=[('STD', StandardScaler()), ('GPR', gp.GaussianProcessRegressor())])
        param_grid={

            'GPR__alpha':[0.03, 0.05, 0.07]

        }

        search = GridSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=self.jobs)
        search.fit(self.x, self.y)
        
        if (self.validate == False):
            return search
        else:
            return search.best_score_
    
    
    def mlp_model(self):
    
        gc.collect()
        pipe = Pipeline(steps=[('STD', StandardScaler()), ('MLP', MLPRegressor())])
        param_grid={
            
            'MLP__hidden_layer_sizes': [(12,4), (10,)],
            'MLP__activation': ['tanh', 'relu'],
            'MLP__solver': ['sgd', 'adam'],
            'MLP__alpha': [0.001, 0.005, 0.01],
            'MLP__learning_rate': ['constant','adaptive'],
            'MLP__early_stopping': [True],
            'MLP__random_state': [self.seed]
        }

        search = GridSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=self.jobs)
        search.fit(self.x, self.y)
     
        if (self.validate == False):
            return search
        else:
            return search.best_score_


    def xgb_model(self):
  
        gc.collect()
        pipe = Pipeline(steps=[('STD', StandardScaler()), ('XGB', XGBRegressor(objective='reg:squarederror'))])
        param_grid={

            'XGB__max_depth': [3, 7],
            
        }

        search = GridSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=self.jobs)
        search.fit(self.x, self.y)
     
        if (self.validate == False):
            return search
        else:
            return search.best_score_
    
    
    def svr_model(self):
        
        gc.collect()
        pipe = Pipeline(steps=[('STD', StandardScaler()), ('SVR', SVR())])
        param_grid={
            
            'SVR__kernel': ['rbf', 'poly'],
            'SVR__C': [1, 50, 100],
            'SVR__epsilon': [0.0001, 0.0005, 0.001]
        }

        search = GridSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=self.jobs)
        search.fit(self.x, self.y)
     
        if (self.validate == False):
            return search
        else:
            return search.best_score_

            
# In[ ]:


# In[ ]:

class preprocessing:

    def __init__(self, df, target='Y', p=7, create_testset=False):
        self.df = df
        self.target = target
        self.p = p #Size of Bucket (e.g., Week = 7 or 5, Lag)
        self.create_testset = create_testset


    def run_univariate(self):
                
        df1 = pd.DataFrame()

        if (self.create_testset == False):
            P = self.p + 1
        else:
            P = self.p

        for i in range(P):
            if (i == 0):
                if (P > self.p):
                    df1['Y'] = self.df[self.target]
                else:
                    df1['X0'] = self.df[self.target]
            else:
                column_name = 'X' + str(i)
                df1[column_name] = self.df[self.target].shift(i)

        return df1


    def create_frequency(self):

        df1 = pd.DataFrame()
        df1 = self.df.groupby(np.arrange(len(self.df))//self.p).sum()
        df1.index = self.df.loc[1::self.p, self.fname]

        return df1

# In[ ]:

class validation:

    def __init__(self, i, X, y, cv=5):
        self.i = i
        self.X = X
        self.y = y
        self.cv = cv

    def ml(self):

        if (self.i == 1):
            score = mlmodels(self.X, self.y, self.cv, True).svr_model()
        elif (self.i == 2):
            score = mlmodels(self.X, self.y, self.cv, True).mlp_model()
        elif (self.i == 3):
            score = mlmodels(self.X, self.y, self.cv, True).xgb_model()
        else:
            score = mlmodels(self.X, self.y, self.cv, True).gpr_model()
        
        return score


    def dl(self):
        
        score = optimization(self.i, self.X, self.y, self.cv, True).run()

        return score

# In[ ]:

class prediction:

    def __init__(self, i, X, y, cv=5):
        self.i = i
        self.X = X
        self.y = y
        self.cv = cv


    def ml(self):

        model = validation(self.i, self.X, self.y, self.cv, False).ml() 
        
        return model

    
    def dl(self):

        model = validation(self.i, self.X, self.y, self.cv, False).dl() 
        
        return model
        

# In[ ]:

class optimization:

    def __init__(self, i, X, y, cv=5, validation=False):
        self.i = i
        self.X = X
        self.y = y
        self.cv = cv
        self.vaidation = validation
 

    def run(self):
        i = self.i
        X = self.X
        y = self.y
        spilt = round((1/self.cv), 2)


        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=spilt, random_state=232)
        size = len(X.columns)

        def get_tuner(m):

            if (m == 1):
                
                def bi_gru_lstm(hp):
                    model = tf.keras.Sequential()
                    #GRU
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=hp.Int('neurons_gru', 4, 10, 1, default=7), return_sequences=True), input_shape=(size, 1)))
                    model.add(tf.keras.layers.BatchNormalization())
                    #LSTM
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hp.Int('neurons_lstm', 4, 10, 1, default=7), return_sequences=False)))
                    model.add(tf.keras.layers.BatchNormalization())
                    #DENSE
                    model.add(tf.keras.layers.Dense(units=hp.Int('neurons_dense', 4, 10, 1, default=7), activation='relu'))
                    model.add(tf.keras.layers.BatchNormalization())
                    #An output layer that makes a single value prediction. 
                    model.add(tf.keras.layers.Dense(1)) 
                    model.compile(optimizer=tf.keras.optimizers.Adam())
                    return model

                tuner = ModelTuner(oracle=kt.oracles.BayesianOptimization(objective=kt.Objective("loss", "min"), max_trials=3), hypermodel=bi_gru_lstm, project_name='gdf_bi_gru_ltsm')


            elif(m == 2):

                def bi_lstm(hp):
                    model = tf.keras.Sequential()
                    #LSTM
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hp.Int('neurons_lstm', 4, 10, 1, default=7), return_sequences=True), input_shape=(size, 1)))
                    model.add(tf.keras.layers.BatchNormalization())
                    #DENSE
                    model.add(tf.keras.layers.Dense(units=hp.Int('neurons_dense', 4, 10, 1, default=7), activation='relu'))
                    model.add(tf.keras.layers.BatchNormalization())
                    #An output layer that makes a single value prediction. 
                    model.add(tf.keras.layers.Dense(1)) 
                    model.compile(optimizer=tf.keras.optimizers.Adam())
                    return model

                tuner = ModelTuner(oracle=kt.oracles.BayesianOptimization(objective=kt.Objective("loss", "min"), max_trials=3), hypermodel=bi_lstm, project_name='gdf_bi_lstm')

            elif(m == 3):
                
                def gru_lstm(hp):
                    model = tf.keras.Sequential()
                    #GRU
                    model.add(tf.keras.layers.GRU(units=hp.Int('neurons_gru', 4, 10, 1, default=7), return_sequences=True), input_shape=(size, 1))
                    model.add(tf.keras.layers.BatchNormalization())
                    #LSTM
                    model.add(tf.keras.layers.LSTM(units=hp.Int('neurons_lstm', 4, 10, 1, default=7), return_sequences=False))
                    model.add(tf.keras.layers.BatchNormalization())
                    #DENSE
                    model.add(tf.keras.layers.Dense(units=hp.Int('neurons_dense', 4, 10, 1, default=7), activation='relu'))
                    model.add(tf.keras.layers.BatchNormalization())
                    #OUTPUT
                    model.add(tf.keras.layers.Dense(1)) 
                    model.compile(optimizer=tf.keras.optimizers.Adam())
                    return model

                tuner = ModelTuner(oracle=kt.oracles.BayesianOptimization(objective=kt.Objective("loss", "min"), max_trials=3), hypermodel=gru_lstm, project_name='gdf_gru_lstm')

            else:

                def lstm(hp):
                    model = tf.keras.Sequential()
                    #LSTM
                    model.add(tf.keras.layers.LSTM(units=hp.Int('neurons_lstm', 4, 10, 1, default=7), return_sequences=True), input_shape=(size, 1))
                    model.add(tf.keras.layers.BatchNormalization())
                    #DENSE
                    model.add(tf.keras.layers.Dense(units=hp.Int('neurons_dense', 4, 10, 1, default=7), activation='relu'))
                    model.add(tf.keras.layers.BatchNormalization())
                    #OUTPUT
                    model.add(tf.keras.layers.Dense(1)) 
                    model.compile(optimizer=tf.keras.optimizers.Adam())
                    return model

                tuner = ModelTuner(oracle=kt.oracles.BayesianOptimization(objective=kt.Objective("loss", "min"), max_trials=3), hypermodel=lstm, project_name='gdf_lstm')
            
            return tuner

        
        def get_name(m):
            
            if (m == 1):
                name = "GDF-BI_GRU_LTSM"
            elif (m == 2):
                name = "GDF-BI_LSTM"
            elif (m == 3):
                name = "GDF-GRU_LSTM"
            else:
                name = "GDF_LSTM"
            
            return name


        get_tuner(i).search(X, y)
        best_hps = get_tuner(i).get_best_hyperparameters()[0]
        model = get_tuner(i).hypermodel.build(best_hps)


        call_back = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, verbose=0)
        ]
    
        history = model.fit(X, y, validation_data=(test_x, test_y), callbacks=call_back, epochs=30, batch_size=32, verbose=0)
        scores = model.evaluate(test_x, test_y, verbose=0)

        if (self.validation == True):
            visualization(history, round((scores[1]*100), 2), get_name(self.i)).disp_fit()
            print(" ")
            print(" ")
            return None
        else:
            return model


class visualization:

    def __init__(self, history, score, x_name, y_name, labels, df=pd.DataFrame(), bins=3, x_means='x_mu', y_means='y_mu', x_std="x_std", y_std="y_std"):
        self.history = history
        self.score = score
        self.x_name = x_name
        self.y_name = y_name
        self.labels = labels
        self.df = df
        self.bins = bins
        self.x_means = df[x_means]
        self.y_means = df[y_means]
        self.x_std = df[x_std]
        self.y_std = df[y_std]
    

    def disp_fit(self):

        print(" ")
        print(" ")
        msg = f"{self.name} achieved a tunned accuracy of {self.score} percent using Distribution Aware Gradient Descent Optimization."  
        print(msg)

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.legend(['training loss', 'validation loss'])
        plt.show()

        return None

    
    def disp_hist(self):

        hist = self.df.hist(bins = self.bins)
        plt.plot(hist)

        return None


    def disp_stack_bar(self):

        fig, ax = plt.subplot()
        ax.bar(self.labels, self.x_means, 0.33, yerr=self.x_std, label=self.x_name)
        ax.bar(self.labels, self.y_means, 0.33, yerr=self.y_std, label=self.y_name)

        ax.set_title(self.title)
        ax.legend()
        plt.show()

        return None


class selection:

    def __init__(self, X, y):
        self.X = X
        self.y = y


    def ml_run(self):
        X = self.X
        y = self.y

        best_score = 100
        best_model = 1

        for i in range(1, 5, 1):
            gc.collect()
            score = validation(i, X, y).ml()
            if (score < best_score):
                best_score = score
                best_model = i

        return best_score, best_model


    def dl_run(self):
        X = self.X
        y = self.y

        best_score = 100
        best_model = 1

        for i in range(1, 5, 1):
            gc.collect()
            score = validation(i, X, y).dl()
            if (score < best_score):
                best_score = score
                best_model = i
        
        return best_score, best_model



class fitting:

    def __init__(self, X, y, T):
        self.X = X
        self.y = y
        self.T = T


    def autots(self):
        ml_score, ml_model = selection(self.X, self.y).ml_run()
        def get_ml_name(m):
            
            if (m == 1):
                name = "SVR"
            elif (m == 2):
                name = "MLP"
            elif (m == 3):
                name = "XGB"
            else:
                name = "GPR"
            
            return name

        dl_score, dl_model = selection(self.X, self.y).dl_run()
        def get_dl_name(m):
            
            if (m == 1):
                name = "GDF-BI_GRU_LTSM"
            elif (m == 2):
                name = "GDF-BI_LSTM"
            elif (m == 3):
                name = "GDF-GRU_LSTM"
            else:
                name = "GDF_LSTM"
            
            return name

        if (ml_score < dl_score):
            print("ML Model Selected: " + get_ml_name(ml_model) + ", MAE: " + str(ml_score))
            yhat = prediction(ml_model, self.X, self.y).ml().predict(self.T)
        else:
            print("DL Model Selected: " + get_dl_name(dl_model) + ", MAE: " + str(dl_score))
            yhat = prediction(dl_model, self.X, self.y).dl().predict(self.T)

        return yhat
    

    def automl(self):

        ml_score, ml_model = selection(self.X, self.y).ml_run()
        def get_name(m):
            
            if (m == 1):
                name = "SVR"
            elif (m == 2):
                name = "MLP"
            elif (m == 3):
                name = "XGB"
            else:
                name = "GPR"
            
            return name
        
        print(" ")
        print("Models Tested: SVR, MLP, XGB and GPR")
        print("ML Model Selected: " + get_name(ml_score) + " MAE: " + str(ml_score))
        print(" ")
        yhat = prediction(ml_model, self.X, self.y).ml().predict(self.T)
        
        return yhat



class execute:
    
    def __init__(self, train, test, lags=3, ml=False):
        self.train = train
        self.test = test
        self.lags = lags
        self.ml = ml

    def frm(self):
    
        train1 = pd.read_excel(self.train)
        test1 = pd.read_excel(self.test)
        train2 = train1.fillna(0)
        test2 = test1.fillna(0)

        for col in train2.columns:
            target = col
            df2 = preprocessing(train2, target, self.lags, False).run_univariate().dropna().reset_index(drop=True)
            T = preprocessing(test2, target, self.lags, True).run_univariate().dropna().reset_index(drop=True)
            y = df2['Y']
            X = df2.loc[:, df2.columns != 'Y']

            if (self.ml == False):
                predictions = fitting(X, y, T).autots()
            else:
                predictions = fitting(X, y, T).automl()

        return predictions



class ModelTuner(kt.Tuner):

    def run_trial(self, trial, x_train, y_train):
        batch_size = 32
        hp = trial.hyperparameters
        model = self.hypermodel.build(trial.hyperparameters)
        epoch_loss_metric = tf.keras.metrics.Mean()
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('opt_learn_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3),
        clipnorm=hp.Float('opt_clipnorm', min_value=0.001, max_value=1.11, step=0.10, default=1.0),
        clipvalue=hp.Float('opt_clipvalue', min_value=1, max_value=5.50, step=0.25, default=5.0))

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

                #Distribution Aware
                if (shapiro_test.pvalue > 0.05):
                    if (lilliefors_test.pvalue < 0.05):
                        loss = huber(real_y, pred_y)
                    else:
                        loss = mse(real_y, pred_y)
                else:
                    loss = huber(real_y, pred_y)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)

            return loss
            
        #Calculate number of batches and define number of epochs per Trial
        num_of_batches = math.floor(len(x_train) / batch_size)
        epochs = 10
        
        #Run the Trial
        for epoch in range(epochs):
            self.on_epoch_begin(trial, model, epoch, logs={})
            for batch in range(num_of_batches):
                n = batch*batch_size
                self.on_batch_begin(trial, model, batch, logs={})
                batch_loss = run_train_step(x_train[n:n+batch_size], y_train[n:n+batch_size])
                print(float(batch_loss))
                self.on_batch_end(trial, model, batch, logs={"loss": float(batch_loss)})
                
            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epoch, logs={"loss": epoch_loss})
            epoch_loss_metric.reset_states()
        