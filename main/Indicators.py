import pandas as pd
import numpy as np
class Indicators:

    @staticmethod  # Make it a static method
    def calculate_rsi(data, period=14):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data['RSI'] = rsi
        return data

    @staticmethod  # Make it a static method
    def calculate_macd(data):
        short_ema = data['Close'].ewm(span=12, adjust=False).mean()
        long_ema = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = short_ema - long_ema
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        return data
    @staticmethod
    def SMA(data):
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        return data

    def EMA(data):
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
        return data


    @staticmethod  # Make it a static method
    def recommend_stock_action(data):
        data = Indicators.calculate_macd(data)
        data = Indicators.calculate_rsi(data)
        data = Indicators.SMA(data)


        data['SMA_200'] = data['Close'].rolling(window=200).mean()

        latest_rsi = data['RSI'].iloc[-1]
        latest_macd = data['MACD'].iloc[-1]
        latest_signal = data['Signal_Line'].iloc[-1]
        latest_sma50 = data['SMA_50'].iloc[-1]
        latest_sma200 = data['SMA_200'].iloc[-1]
        latest_ema50 = data['EMA_50'].iloc[-1]
        latest_ema200 = data['EMA_200'].iloc[-1]

        buy_signals, sell_signals = [], []

        if latest_sma50 > latest_sma200:
            buy_signals.append("SMA_50 above SMA_200 (Golden Cross)")
        elif latest_sma50 < latest_sma200:
            sell_signals.append("SMA_50 below SMA_200 (Death Cross)")

        if latest_ema50 > latest_ema200:
            buy_signals.append("EMA_50 above EMA_200 (Golden Cross)")
        elif latest_ema50 < latest_ema200:
            sell_signals.append("EMA_50 below EMA_200 (Death Cross)")

        if latest_rsi < 30:
            buy_signals.append("RSI below 30 (Oversold)")
        elif latest_rsi > 70:
            sell_signals.append("RSI above 70 (Overbought)")

        if latest_macd > latest_signal:
            buy_signals.append("MACD above Signal Line")
        elif latest_macd < latest_signal:
            sell_signals.append("MACD below Signal Line")

        if len(buy_signals) > len(sell_signals):
            return f"**Recommendation: BUY** 📈/nReasons: {', '.join(buy_signals)}"
        elif len(sell_signals) > len(buy_signals):
            return f"**Recommendation: SELL** 📉/nReasons: {', '.join(sell_signals)}"
        else:
            return "**Recommendation: HOLD** 🤔/nMarket is neutral or mixed signals."
        # _______________________________________________________________________________________
        # data = Indicators.calculate_macd(data)
        # data = Indicators.calculate_rsi(data)
        # data = Indicators.SMA(data)
        # # data = Indicators.EMA(data)

        # data['SMA_200'] = data['Close'].rolling(window=200).mean()

        # latest_rsi = data['RSI'].dropna().iloc[-1]  # Ensuring no NaN values
        # latest_macd = data['MACD'].iloc[-1]
        # latest_signal = data['Signal_Line'].iloc[-1]
        # latest_sma50 = data['SMA_50'].iloc[-1]
        # latest_sma200 = data['SMA_200'].iloc[-1]
        # latest_ema50 = data['EMA_50'].iloc[-1]
        # latest_ema200 = data['EMA_200'].iloc[-1]

        # buy_signals, sell_signals = [], []

        # # Checking SMA crossover
        # if data['SMA_50'].iloc[-2] < data['SMA_200'].iloc[-2] and latest_sma50 > latest_sma200:
        #     buy_signals.append("SMA_50 crossed above SMA_200 (Golden Cross)")
        # elif data['SMA_50'].iloc[-2] > data['SMA_200'].iloc[-2] and latest_sma50 < latest_sma200:
        #     sell_signals.append("SMA_50 crossed below SMA_200 (Death Cross)")

        # # Checking EMA crossover
        # if data['EMA_50'].iloc[-2] < data['EMA_200'].iloc[-2] and latest_ema50 > latest_ema200:
        #     buy_signals.append("EMA_50 crossed above EMA_200 (Golden Cross)")
        # elif data['EMA_50'].iloc[-2] > data['EMA_200'].iloc[-2] and latest_ema50 < latest_ema200:
        #     sell_signals.append("EMA_50 crossed below EMA_200 (Death Cross)")

        # # RSI condition
        # if latest_rsi < 30:
        #     buy_signals.append("RSI below 30 (Oversold)")
        # elif latest_rsi > 70:
        #     sell_signals.append("RSI above 70 (Overbought)")

        # # MACD condition
        # if latest_macd > latest_signal:
        #     buy_signals.append("MACD above Signal Line")
        # elif latest_macd < latest_signal:
        #     sell_signals.append("MACD below Signal Line")

        # # Decision logic
        # if len(buy_signals) > len(sell_signals):
        #     return f"**Recommendation: BUY** 📈\nReasons: {', '.join(buy_signals)}"
        # elif len(sell_signals) > len(buy_signals):
        #     return f"**Recommendation: SELL** 📉\nReasons: {', '.join(sell_signals)}"
        # else:
        #     return "**Recommendation: HOLD** 🤔\nNo strong signals detected."




