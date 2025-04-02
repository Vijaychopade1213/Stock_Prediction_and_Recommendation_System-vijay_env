import yfinance as yf
import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.metrics import mean_absolute_percentage_error 
from xgboost import XGBRegressor 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, GRU, Dense 
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.subplots as sp
import requests
import pickle
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request,session
import os
class Predictions:
    def predict_current_price(self, data, ml_model, xgb_model, dl_model, scaler):
        latest_data = data[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)

        # ML Prediction
        ml_prediction = ml_model.predict(latest_data)[0]

        # XGB Prediction
        xgb_prediction = xgb_model.predict(latest_data)[0]

        # DL Prediction
        latest_data_dl = np.reshape(latest_data, (latest_data.shape[0], 1, latest_data.shape[1]))
        dl_prediction = dl_model.predict(latest_data_dl)[0][0]

        original_min = scaler.data_min_[3]  
        original_max = scaler.data_max_[3]  

        ml_actual_price = ml_prediction * (original_max - original_min) + original_min
        xgb_actual_price = xgb_prediction * (original_max - original_min) + original_min
        dl_actual_price = dl_prediction * (original_max - original_min) + original_min

        print(f"ML Prediction: {ml_actual_price}")
        print(f"XGB Prediction: {xgb_actual_price}")
        print(f"DL Prediction: {dl_actual_price}")


        aggregated_price = (ml_actual_price + xgb_actual_price + dl_actual_price) / 3
        print(f"Aggregated Predicted Price: {aggregated_price}")

        return ml_actual_price, xgb_actual_price, dl_actual_price,aggregated_price

