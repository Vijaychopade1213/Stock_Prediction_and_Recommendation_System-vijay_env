import yfinance as yf
import pandas as pd
import numpy as np
import requests
# class StockDataFetcher:
#     def fetch_stock_data(self, symbol, period='10y'):
        
#         stock = yf.Ticker(symbol)
#         stock.info 
#         data = stock.history(period=period)
#         info = stock.info

# # Display relevant stock details
#         print(f"Company Name: {info.get('longName', 'N/A')}")
#         print(f"Stock Price: {info.get('currentPrice', 'N/A')}")
#         print(f"Market Cap: {info.get('marketCap', 'N/A')}")
#         print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
#         print(f"Sector: {info.get('sector', 'N/A')}")
#         print(f"Industry: {info.get('industry', 'N/A')}")
#         return data,info

#     def fetch_live_data(self, symbol, api_key):
#         api_key="771O3VPDZ5UH78E3"
#         """Fetches live stock data using Alpha Vantage API."""
#         try:
#             url = 'https://www.alphavantage.co/query'
#             params = {
#                 "function": "TIME_SERIES_DAILY",
#                 "symbol": symbol,
#                 "apikey": api_key,
#                 "outputsize": "compact" # 
#             }
#             response = requests.get(url, params=params)
#             response.raise_for_status()  
#             data = response.json()

#             if "Time Series (Daily)" in data:
#                 # Convert the data to a pandas DataFrame
#                 df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
#                 df.index = pd.to_datetime(df.index)
#                 df.sort_index(inplace=True)  # Sort by date
#                 df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#                 df = df.astype(float)  # Convert columns to numeric

#                 return df
#             else:
#                 print(f"Error fetching live data for {symbol}: {data.get('Error Message', 'No error message provided')}")
#                 return pd.DataFrame()
#         except requests.exceptions.RequestException as e:
#             print(f"Request failed for {symbol}: {e}")
#             return pd.DataFrame()
#         except (ValueError, KeyError) as e:
#             print(f"Error parsing live data for {symbol}: {e}")
#             return pd.DataFrame()


#     def combine_data(self, historical_data, live_data):
#         if historical_data.empty:
#             return live_data
#         if live_data.empty:
#             return historical_data

#         combined_data = pd.concat([historical_data, live_data])
#         combined_data = combined_data[~combined_data.index.duplicated(keep='first')]  

#
# 
class StockDataFetcher:
    def fetch_stock_data(self, symbol, period='20y'):
        
        stock = yf.Ticker(symbol)
        stock.info 
        data = stock.history(period=period)
        if symbol.startswith('^'):
            info = stock.info
            period = '1h'
        else:
            info = stock.info
            period = period

# Display relevant stock details
        print(f"Company Name: {info.get('longName', 'N/A')}")
        print(f"Stock Price: {info.get('currentPrice', 'N/A')}")
        print(f"Market Cap: {info.get('marketCap', 'N/A')}")
        print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Industry: {info.get('industry', 'N/A')}")
        return data,info

    def fetch_live_data(self, symbol, api_key):
        api_key="GHIJOQOQJ95KDBPK"
        """Fetches live stock data using Alpha Vantage API."""
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": api_key,
                "outputsize": "compact" # 
            }
            response = requests.get(url, params=params)
            response.raise_for_status()  
            data = response.json()

            if "Time Series (Daily)" in data:
                # Convert the data to a pandas DataFrame
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)  # Sort by date
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df.astype(float)  # Convert columns to numeric

                return df
            else:
                print(f"Error fetching live data for {symbol}: {data.get('Error Message', 'No error message provided')}")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {symbol}: {e}")
            return pd.DataFrame()
        except (ValueError, KeyError) as e:
            print(f"Error parsing live data for {symbol}: {e}")
            return pd.DataFrame()


    def combine_data(self, historical_data, live_data):
        if historical_data.empty:
            return live_data
        if live_data.empty:
            return historical_data
        

        combined_data = pd.concat([historical_data, live_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]  
    


        return combined_data.sort_index()
