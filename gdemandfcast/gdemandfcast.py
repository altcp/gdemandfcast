#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import gc
import math
import arch as am
import numpy as np
import pandas as pd
import pmdarima as pm
import tensorflow as tf
import kerastuner as kt
import sklearn.gaussian_process as gp

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from pmdarima.arima import ndiffs

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
    
    def __init__(self, y, D, P, Q, seed, seasonal=True, alpha=0.05):
        self.y = y
        self.D = D
        self.P = P
        self.Q = Q
        self.seasonal = seasonal
        self.seed = seed
        self.alpha = alpha
    
    
    def run_model(self):
        
        kpss_test = ndiffs(self.y, alpha=self.alpha, test='kpss', max_d=self.D)
        adf_test = ndiffs(self.y, alpha=self.alpha, test='adf', max_d=self.D)
        num_of_diffs = max(kpss_test, adf_test)
    
        arima_model = pm.auto_arima(self.y, d=num_of_diffs, start_p=0, start_q=0, start_P=0, max_p=self.P, max_q=self.Q, trace=False, 
                                    seasonal=self.seasonal, error_action='ignore', random_state=self.seed, suppress_warnings=True)
        
        
        e_mu = arima_model.predict(n_periods=1)
        e_mu = e_mu[0]
        
        return e_mu

# In[ ]:





# In[ ]:

class mlmodels:
    
    def __init__(self, X, y, cv=5, scoring='r2', num_of_cpu=-2, seed=232, use_tensorflow=False):
        self.x = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.jobs = num_of_cpu
        self.seed = seed
        self.use_tensorflow = use_tensorflow
    
    
    def gpr_model(self):
        
        if(self.use_tensorflow != True):
    
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
        
        else:
            
            print("ERR. Probabilistic Modeling.")
            
     
        return search
    
    
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
     
        return search
    
    
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
     
        return search

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
        def run_train_step(real_x, real_y, use_huber):
            with tf.GradientTape() as tape:
                pred_y = model(real_x)

                if (use_huber == True):
                    loss=tf.keras.losses.MSE(real_y, pred_y)
                else:
                    dev = []
                    dev = abs(real_y - pred_y)
                    q3, q1 = np.percentile(dev, [75, 25])
                    iqr = q3 -q1
                    delta = q3 + (1.5 * iqr)
                    t = tf.norm((real_y -pred_y), ord=1) / len(real_y)
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

    def __init__(self, df, target='Y', p=3, create_testset=False , from_excel=False, location=" "):
        self.df = df
        self.target = target
        self.p = p
        self.create_testset = create_testset
        self.from_excel = from_excel
        self.location = location

    def run_prep(self):
        
        if (self.from_excel == False):
            df1 = pd.DataFrame()
        else:
            df1 = pd.DataFrame()
            #Soon.

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
# In[ ]:





# In[ ]:





# In[ ]:




