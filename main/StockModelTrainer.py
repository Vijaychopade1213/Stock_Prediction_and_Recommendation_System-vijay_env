import yfinance as yf
import pandas as pd
import numpy as np
import joblib  # For loading saved models
from sklearn.model_selection import train_test_split  #For Model Training
from sklearn.ensemble import RandomForestRegressor  #For Model Training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #For Model Training
from sklearn.metrics import mean_absolute_percentage_error #For Model Training
from xgboost import XGBRegressor #For Model Training
from tensorflow.keras.models import Sequential #For Model Training
from tensorflow.keras.layers import LSTM, GRU, Dense #For Model Training
from sklearn.preprocessing import MinMaxScaler

class StockModelTrainer: #These are not nesscary for running the APP but i leave it here to train the model
    def train_ml_model(self, df):
        X, y = df[['Open', 'High', 'Low', 'Volume']], df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print("Random Forest Model - Accuracy Metrics:")
        print(f"R-squared: {r2 * 100:.2f}%")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")
        return model

    def train_xgb_model(self, df):
        X, y = df[['Open', 'High', 'Low', 'Volume']], df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print("XGBoost Model - Accuracy Metrics:")
        print(f"R-squared: {r2 * 100:.2f}%")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")
        return model

    def train_dl_model(self, df):
        X = df[['Open', 'High', 'Low', 'Volume']].values.reshape(df.shape[0], 1, 4)
        y = df['Close'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 4)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print("Deep Learning Model - Accuracy Metrics:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")
        return model
