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


class StockVisualization:
    def create_candlestick_chart(self, data, symbol):
      
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'])])

        fig.update_layout(
            title=f'{symbol} Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark' 
        )

        return fig.to_html(full_html=False)  

    def generate_charts(self,data, symbol):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
                  x=data.index,
                  y=data['Close'],
                  mode='lines',
                  name='Close Price',
                  line=dict(color='blue')
              ))

             
        fig.update_layout(
                  title=f'{symbol} Stock Price',
                  xaxis_title='Date',
                  yaxis_title='Close Price',
                  template='plotly_dark', )

              # Show the plot
        #fig.show()
        return fig.to_html(full_html=False)

    # Create subplots: 2 rows, 2 columns
    def generate_all_charts(self,data, symbol):
        fig = sp.make_subplots(rows=2, cols=2,
                              subplot_titles=(f'{symbol} Stock Price with SMA',
                                              f'{symbol} Stock Price with EMA',
                                              'Relative Strength Index',
                                              'MACD Indicator'),
                              vertical_spacing=0.2)

        # Close Price and SMA (50 & 200)
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='200-Day SMA', line=dict(color='green')), row=1, col=1)

        # EMA (50 & 200)
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='50-Day EMA', line=dict(color='purple')), row=1, col=2)
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], mode='lines', name='200-Day EMA', line=dict(color='orange')), row=1, col=2)

        # RSI with Overbought (70) and Oversold (30) Levels
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='brown')), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')), row=2, col=1)

        # MACD and Signal Line (if MACD is present)
        if 'MACD' in data.columns and 'Signal_Line' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='yellow')), row=2, col=2)
            fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='red', dash='dash')), row=2, col=2)

        # Update layout
        fig.update_layout(title=f'{symbol} Stock Analysis',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          height=1000, width=1450,
                          showlegend=True)

        # Show the interactive plot
        #fig.show()
        return fig.to_html(full_html=False)

    def generate_indicator_chart(self, data, symbol, indicator):
        fig = go.Figure()

        if indicator == 'rsi':
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
            fig.add_trace(go.Scatter(x=data.index, y=[70] * len(data), mode='lines', name='Overbought (70)', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=data.index, y=[30] * len(data), mode='lines', name='Oversold (30)', line=dict(dash='dash')))
            fig.update_layout(title=f'{symbol} - RSI', yaxis_title='RSI Value')
        elif indicator == 'macd':
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
            fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line', line=dict(dash='dash')))
            fig.update_layout(title=f'{symbol} - MACD', yaxis_title='MACD Value')
        elif indicator == 'sma':
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200'))
            fig.update_layout(title=f'{symbol} - SMA', yaxis_title='Price')
        elif indicator == 'ema':
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50'))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], mode='lines', name='EMA 200'))
            fig.update_layout(title=f'{symbol} - EMA', yaxis_title='Price')

        return fig.to_html(full_html=False)
    