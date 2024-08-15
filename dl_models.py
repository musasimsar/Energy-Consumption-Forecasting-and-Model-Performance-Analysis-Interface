import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DLModels:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.ann_model = None
        self.lstm_model = None
    
    def build_ann(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.X_train.shape[1], activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def build_lstm(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.X_train.shape[1], 1), return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def evaluate(self, model):
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        return mse, r2
    
    def ann(self, epochs, batch_size):
        self.ann_model = self.build_ann()
        self.ann_model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        return self.evaluate(self.ann_model)
    
    def lstm(self, epochs, batch_size):
        self.X_train = np.array(self.X_train).reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = np.array(self.X_test).reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
        self.lstm_model = self.build_lstm()
        self.lstm_model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        return self.evaluate(self.lstm_model)
