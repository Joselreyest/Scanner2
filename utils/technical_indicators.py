import pandas as pd
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df):
        """Add common technical indicators to dataframe"""
        df = TechnicalIndicators.add_sma(df, 20)
        df = TechnicalIndicators.add_sma(df, 50)
        df = TechnicalIndicators.add_rsi(df, 14)
        return df
    
    @staticmethod
    def add_sma(df, window):
        """Add Simple Moving Average"""
        df[f'SMA{window}'] = df['Close'].rolling(window=window).mean()
        return df
    
    @staticmethod
    def add_rsi(df, window=14):
        """Add Relative Strength Index"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """Add MACD indicator"""
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        return df