import yfinance as yf
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler



class StockAnalysis:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        df = self.data.copy()
        df.index = pd.to_datetime(df.index)

        for col in df.select_dtypes(include=[np.number]).columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            upper_limit, lower_limit = q3 + 1.5 * iqr, q1 - 1.5 * iqr
            df[col] = df[col].clip(lower_limit, upper_limit)
        
        original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df['Pct_Change'] = df['Close'].pct_change() * 100
        df_display = df[original_columns + ['Pct_Change']]

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        scaler = MinMaxScaler()
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
        return df, scaler,df_display