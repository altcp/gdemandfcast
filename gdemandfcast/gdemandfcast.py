#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import gc
import math
import xgboost
import warnings
import arch as am

import pandas as pd
import pmdarima as pm
import keras_tuner as kt
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import sklearn.gaussian_process as gp

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from scipy import stats

warnings.filterwarnings("ignore")

# In[ ]:





# In[ ]:

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
                
            model = am.arch_model(y=self.y, mean=self.mu, vol=self.vol, dist=self.dist, p=self.p, o=self.o, q=self.q, power=self.power, rescale=True)
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
    
    def __init__(self, X, y, cv=5, scoring='r2', num_of_cpu=-2, seed=232, validate=False):
        self.x = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.jobs = num_of_cpu
        self.seed = seed
        self.validate = validate
    
    
    def gpr_model(self):
        
        gc.collect() 
        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-1, 1e3))
        pipe = Pipeline(steps=[('STD', StandardScaler()), ('GPR', gp.GaussianProcessRegressor())])
        param_grid={

            'GPR__kernel':[kernel],
            'GPR__n_restarts_optimizer': [3, 5, 7],
            'GPR__alpha':[0.03, 0.05, 0.07],
            'GPR__random_state': [self.seed]
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
            
            'MLP__hidden_layer_sizes': [(9,6,3),(3,3,3),(12,4),(30,),(10,)],
            'MLP__activation': ['tanh', 'relu'],
            'MLP__solver': ['sgd', 'adam'],
            'MLP__alpha': [0.0005, 0.001, 0.005, 0.01, 0.05],
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
            
            'XGB__n_estimators': [100, 300, 500],
            'XGB__max_depth': [3, 6, 9],
            'XGB__verbosity': [0],
            'XGB__alpha': [0, 0.003, 0.005, 0.01, 0.03],
            'XGB__eta': [0.05, 0.1, 0.3, 0.5, 0.7, 1],
            'XGB__subsample': [0.5, 0.7, 1],
            'XGB__gamma': [0, 3, 6, 9],
            'XGB__random_state': [self.seed]
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

class dlmodels:
    
    def __init__(self, feature_size, min_units=4, max_units=10, step_units=1, units=7):
        self.min_units = min_units
        self.max_units = max_units
        self.step_units = step_units
        self.units = units
        self.feature_size = feature_size

        
    def bi_gru_lstm(self, hp):
    
        model = tf.keras.Sequential()

        #GRU
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=hp.Int('neurons_gru', self.min_units, self.max_units, self.step_units, default=self.units), 
                                                                    input_shape=(self.feature_size, 1), activation='relu', 
                                                                    recurrent_dropout=hp.Float('rcc_dropout_gru', min_value=0.0, max_value=0.4, step=0.2, default=0.2),
                                                                    return_sequences=True)))
        model.add(tf.keras.layers.BatchNormalization())
    

        #LSTM
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hp.Int('neurons_lstm', self.min_units, self.max_units, self.step_units, default=self.units), 
                                                                     activation='relu', recurrent_dropout=hp.Float('rcc_dropout_lstm', min_value=0.0, max_value=0.4, 
                                                                     step=0.2, default=0.2), return_sequences=False)))
        model.add(tf.keras.layers.BatchNormalization())

    
        #DENSE
        model.add(tf.keras.layers.Dense(units=hp.Int('neurons_dense', self.min_units, self.max_units, self.step_units, default=self.units), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
    
        #An output layer that makes a single value prediction. 
        model.add(tf.keras.layers.Dense(1)) 
     
        #Tune the optimizer's learning rate.
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp.Float('opt_learn_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3),
                                                         clipnorm=hp.Float('opt_clipnorm', min_value=0.001, max_value=1.11, step=0.10, default=1.0),
                                                         clipvalue=hp.Float('opt_clipvalue', min_value=1, max_value=5.50, step=0.25, default=5.0)), 
                                                         loss='mse', metrics=['mae'])
        return model
    
    
    def gru_lstm(self, hp):
    
        model = tf.keras.Sequential()

        #GRU
        model.add(tf.keras.layers.GRU(units=hp.Int('neurons_gru', self.min_units, self.max_units, self.step_units, default=self.units), 
                                      input_shape=(self.feature_size, 1), activation='relu', 
                                      recurrent_dropout=hp.Float('rcc_dropout_gru', min_value=0.0, max_value=0.4, step=0.2, default=0.2),
                                      return_sequences=True))
        
        model.add(tf.keras.layers.BatchNormalization())
    

        #LSTM
        model.add(tf.keras.layers.LSTM(units=hp.Int('neurons_lstm', self.min_units, self.max_units, self.step_units, default=self.units), 
                                       activation='relu', recurrent_dropout=hp.Float('rcc_dropout_lstm', min_value=0.0, max_value=0.4, 
                                                                                     step=0.2, default=0.2), return_sequences=False))
        model.add(tf.keras.layers.BatchNormalization())

    
        #DENSE
        model.add(tf.keras.layers.Dense(units=hp.Int('neurons_dense', self.min_units, self.max_units, self.step_units, default=self.units), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
    
        #An output layer that makes a single value prediction. 
        model.add(tf.keras.layers.Dense(1)) 
     
        #Tune the optimizer's learning rate.
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp.Float('opt_learn_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3),
                                                         clipnorm=hp.Float('opt_clipnorm', min_value=0.001, max_value=1.11, step=0.10, default=1.0),
                                                         clipvalue=hp.Float('opt_clipvalue', min_value=1, max_value=5.50, step=0.25, default=5.0)), 
                                                         loss='mse', metrics=['mae'])
        return model
    
    
    def bi_lstm(self, hp):
    
        model = tf.keras.Sequential()

        #LSTM
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hp.Int('neurons_lstm', self.min_units, self.max_units, self.step_units, default=self.units), 
                                                                     activation='relu', recurrent_dropout=hp.Float('rcc_dropout_lstm', min_value=0.0, max_value=0.4, 
                                                                     step=0.2, default=0.2), return_sequences=False)))
        
        model.add(tf.keras.layers.BatchNormalization())

    
        #DENSE
        model.add(tf.keras.layers.Dense(units=hp.Int('neurons_dense', self.min_units, self.max_units, self.step_units, default=self.units), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
    
        #An output layer that makes a single value prediction. 
        model.add(tf.keras.layers.Dense(1)) 
     
        #Tune the optimizer's learning rate.
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp.Float('opt_learn_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3),
                                                         clipnorm=hp.Float('opt_clipnorm', min_value=0.001, max_value=1.11, step=0.10, default=1.0),
                                                         clipvalue=hp.Float('opt_clipvalue', min_value=1, max_value=5.50, step=0.25, default=5.0)), 
                                                         loss='mse', metrics=['mae'])
        return model
    

    def lstm(self, hp):
    
        model = tf.keras.Sequential()

        #LSTM
        model.add(tf.keras.layers.LSTM(units=hp.Int('neurons_lstm', self.min_units, self.max_units, self.step_units, default=self.units), 
                                       activation='relu', recurrent_dropout=hp.Float('rcc_dropout_lstm', min_value=0.0, max_value=0.4, 
                                                                                     step=0.2, default=0.2), return_sequences=False))
        
        model.add(tf.keras.layers.BatchNormalization())

        #DENSE
        model.add(tf.keras.layers.Dense(units=hp.Int('neurons_dense', self.min_units, self.max_units, self.step_units, default=self.units), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
    
        #An output layer that makes a single value prediction. 
        model.add(tf.keras.layers.Dense(1)) 
     
        #Tune the optimizer's learning rate.
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp.Float('opt_learn_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3),
                                                         clipnorm=hp.Float('opt_clipnorm', min_value=0.001, max_value=1.11, step=0.10, default=1.0),
                                                         clipvalue=hp.Float('opt_clipvalue', min_value=1, max_value=5.50, step=0.25, default=5.0)), 
                                                         loss='mse', metrics=['mae'])
        return model

# In[ ]:





# In[ ]:

class ModelTuner(kt.Tuner):

    def run_trial(self, trial, x_train, y_train, batch_size):
        hp = trial.hyperparameters
        model = self.hypermodel.build(trial.hyperparameters)
        epoch_loss_metric = tf.keras.metrics.Mean()
        optimizer=tf.keras.optimizers.Adam(lr=hp.Float('opt_learn_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3),
                                                     clipnorm=hp.Float('opt_clipnorm', min_value=0.001, max_value=1.11, step=0.10, default=1.0),
                                                     clipvalue=hp.Float('opt_clipvalue', min_value=1, max_value=5.50, step=0.25, default=5.0))

        @tf.function
        def run_train_step(real_x, real_y, alpha=0.05):
            with tf.GradientTape() as tape:
                pred_y = model(real_x)

                data = []
                data = real_y - pred_y
                shapiro_test = stats.shapiro(data)

                #Distribution Aware
                if (shapiro_test.pvalue > alpha):
                    loss=tf.keras.losses.MSE(real_y, pred_y)
                else:
                    dev = []
                    dev = abs(real_y - pred_y)
                    q3, q1 = np.percentile(dev, [75, 25])
                    iqr = q3 - q1
                    delta = q3 + (1.5 * iqr)
                    t = tf.norm((real_y - pred_y), ord=1) / len(real_y)
                    loss = ((delta * t) - (0.5 * (delta**2)))

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
                self.on_batch_end(trial, model, batch, logs={'loss': batch_loss})
                
        epoch_loss = epoch_loss_metric.result().numpy()
        self.on_epoch_end(trial, model, epoch, logs={'loss': epoch_loss})
        epoch_loss_metric.reset_states()

# In[ ]:

class preprocessing:

    def __init__(self, df, target='Y', p=7, create_testset=False , from_excel=" ", sheet_name=0, fname="Weekly"):
        self.df = df
        self.target = target
        self.p = p #Size of Bucket (e.g., Week = 7 or 5, Lag)
        self.create_testset = create_testset
        self.from_excel = from_excel
        self.sheet_name = sheet_name
        self.fname = fname

    def run_univariate(self):
        
        if (self.from_excel == " "):
            df1 = pd.DataFrame()
        else:
            df1 = pd.DataFrame()
            df1 = pd.read_excel(self.from_excel, self.sheet_name)

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

    def __init__(self, i, X, y, scoring, cv, val=True):
        self.i = i
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        self.val  = val

    def ml(self):
        return {
            1 : mlmodels(self.X, self.y, self.cv, self.scoring, 3, 232, self.val).svr_model(),
            2 : mlmodels(self.X, self.y, self.cv, self.scoring, 3, 232, self.val).mlp_model(),
            3 : mlmodels(self.X, self.y, self.cv, self.scoring, 3, 232, self.val).xgb_model(),
            4 : mlmodels(self.X, self.y, self.cv, self.scoring, 3, 232, self.val).gpr_model()
        } [self.i] 


    def dl(self):
        return {
            1 : optimization(1, self.X, self.y, self.cv, 232, self.val).run(),
            2 : optimization(2, self.X, self.y, self.cv, 232, self.val).run(),
            3 : optimization(3, self.X, self.y, self.cv, 232, self.val).run(),
            4 : optimization(4, self.X, self.y, self.cv, 232, self.val).run()
        } [self.i] 


# In[ ]:

class prediction:

    def __init__(self, i, X, y, T, scoring='mean_absolute_error', cv=5):
        self.i = i
        self.X = X
        self.y = y
        self.T = T
        self.scoring = scoring
        self.cv = cv


    def ml(self):
        return {
            1 : validation(1, self.X, self.y, self.cv, self.scoring, 3, 232, False).ml().predict(self.T), #svr
            2 : validation(2, self.X, self.y, self.cv, self.scoring, 3, 232, False).ml().predict(self.T), #mlp
            3 : validation(3, self.X, self.y, self.cv, self.scoring, 3, 232, False).ml().predict(self.T), #xgb
            4 : validation(4, self.X, self.y, self.cv, self.scoring, 3, 232, False).ml().predict(self.T)  #gpr
        } [self.i] 

    
    def dl(self):
        return {
            1 : validation(1, self.X, self.y, self.cv, 'min', 3, 232, False).dl().predict(self.T), #lstm
            2 : validation(2, self.X, self.y, self.cv, 'min', 3, 232, False).dl().predict(self.T), #bi_lstm
            3 : validation(3, self.X, self.y, self.cv, 'min', 3, 232, False).dl().predict(self.T), #gru_lstm
            4 : validation(4, self.X, self.y, self.cv, 'min', 3, 232, False).dl().predict(self.T)  #bi_gru_lstm
        } [self.i]
        

# In[ ]:

class optimization:

    def __init__(self, i, X, y, cv=5, scoring='min', cpusize=3, seed=232, validation=False, batchsize=32, epoch=30):
        self.i = i
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.cpusize = cpusize
        self.seed = seed
        self.vaidation = validation
        self.batchsize = batchsize
        self.epoch = epoch


    def run(self):
        i = self.i
        X = self.X
        y = self.y
        spilt = round((1/self.cv), 2)
        scoring = self.scoring
        cpusize = self.cpusize
        batchsize = self.batchsize
        epoch = self.epoch
        seed = self.seed


        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=spilt, random_state=seed)


        def get_tuner(m):
            return {
                1: ModelTuner(oracle = kt.oracles.BayesianOptimization(objective=kt.Objective('loss', scoring), max_trials=cpusize, seed=seed), hypermodel=dlmodels().lstm(), project_name='gdf_lstm'),
                2: ModelTuner(oracle = kt.oracles.BayesianOptimization(objective=kt.Objective('loss', scoring), max_trials=cpusize, seed=seed), hypermodel=dlmodels().bi_lstm(), project_name='gdf_bi_lstm'),
                3: ModelTuner(oracle = kt.oracles.BayesianOptimization(objective=kt.Objective('loss', scoring), max_trials=cpusize, seed=seed), hypermodel=dlmodels().gru_lstm(), project_name='gdf_gru_lstm'),
                4: ModelTuner(oracle = kt.oracles.BayesianOptimization(objective=kt.Objective('loss', scoring), max_trials=cpusize, seed=seed), hypermodel=dlmodels().bi_gru_lstm(), project_name='gdf_bi_gru_lstm')
            } [self.i]

        
        def get_name(m):
            return {
                1: "GDF-LTSM",
                2: "GDF-BI_LSTM",
                3: "GDF-GRU_LSTM",
                4: "GDF-BI-GRU_LSTM"
            } [self.i]
    

        get_tuner(i).search(X, y)
        best_hps = get_tuner(i).get_best_hyperparameters()[0]
        model = get_tuner(i).hypermodel.build(best_hps)

        call_back = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, verbose=0)
        ]
    
        history = model.fit(X, y, validation_data=(test_x, test_y), callbacks=call_back, epochs=epoch, batch_size=batchsize, verbose=0)
        scores = model.evaluate(test_x, test_y, verbose=0)

        if (self.validation == True):
            visualization(history, round((scores[1]*100), 2), get_name(self.i)).disp_fit()
            return None
        else:
            return model


class visualization:

    def __init__(self, history, score, name, df=pd.DataFrame(), bins=3, x_target='X', y_target='y'):
        self.history = history
        self.score = score
        self.name = name
        self.df = df
        self.bins = bins
        self.x_mean = df[x_target].mean()
        self.y_mean = df[y_target].mean()
        self.x_std = df[x_target].std()
        self.y_std = df[y_target].std()
    

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
