import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from io import StringIO
import numpy as np
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('scanner_debug.log')  # File output
    ]
)

# Initialize debug statistics
debug_stats = defaultdict(int)

# Initialize technical indicators with improved calculation
class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df, min_periods=20):
        """Add technical indicators with proper period checks"""
        try:
            # Ensure we have enough data for the longest indicator (SMA50)
            if len(df) < min_periods:
                return df
            
            # Calculate SMAs only if we have enough data
            if len(df) >= 20:
                df['SMA20'] = df['Close'].rolling(window=20, min_periods=20).mean()
            if len(df) >= 50:
                df['SMA50'] = df['Close'].rolling(window=50, min_periods=50).mean()
            
            # Calculate RSI only if we have enough data
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
            
            return df
        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            return df

tech = TechnicalIndicators()

# Set page config
st.set_page_config(
    page_title="Stock Scanner Pro with Fixed SMA",
    page_icon="📈",
    layout="wide"
)

# [Previous functions like get_sp500_symbols(), get_nasdaq_symbols(), etc. remain the same]

def get_breakouts(min_price, min_volume, symbols, max_symbols_to_scan):
    """Scan for breakout stocks with proper SMA handling"""
    results = []
    debug_reasons = defaultdict(list)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    symbols_to_scan = symbols[:max_symbols_to_scan]
    total_symbols = len(symbols_to_scan)
    
    for i, symbol in enumerate(symbols_to_scan):
        try:
            status_text.text(f"Scanning {symbol} ({i+1}/{total_symbols})")
            progress_bar.progress((i+1)/total_symbols)
            
            stock = yf.Ticker(symbol)
            
            # Get sufficient historical data for indicators
            hist = stock.history(period="3mo")  # Increased from 1mo to ensure enough data points
            
            # Skip if not enough data
            if len(hist) < 50:  # Need at least 50 days for SMA50
                debug_reasons["Insufficient historical data"].append(symbol)
                log_debug(symbol, f"Skipped - Only {len(hist)} days of data, need at least 50")
                continue
                
            if hist.empty:
                debug_reasons["Empty history"].append(symbol)
                continue
                
            # Add technical indicators
            hist = tech.add_all_indicators(hist)
            
            # Get current values
            current_price = hist['Close'][-1]
            current_volume = hist['Volume'][-1]
            avg_volume = hist['Volume'].mean()
            recent_high = hist['High'][-5:-1].max()  # High of last 5 days excluding today
            
            # Check if indicators were calculated
            sma20_available = 'SMA20' in hist and not pd.isna(hist['SMA20'][-1])
            sma50_available = 'SMA50' in hist and not pd.isna(hist['SMA50'][-1])
            
            # Check each condition separately with proper validation
            conditions = {
                'Price too low': current_price <= min_price,
                'Volume too low': current_volume <= min_volume,
                'Not breaking out': current_price <= recent_high,
                'SMA20 not available': not sma20_available,
                'SMA50 not available': not sma50_available,
                'SMA crossover invalid': sma20_available and sma50_available and hist['SMA20'][-1] <= hist['SMA50'][-1]
            }
            
            rejected = False
            for reason, condition in conditions.items():
                if condition:
                    debug_reasons[reason].append(symbol)
                    log_debug(symbol, f"Breakout Rejected - {reason}")
                    rejected = True
                    break
            
            if not rejected:
                results.append({
                    'Symbol': symbol,
                    'Price': current_price,
                    'Breakout Level': recent_high,
                    'Volume': current_volume,
                    'SMA20': hist['SMA20'][-1] if sma20_available else None,
                    'SMA50': hist['SMA50'][-1] if sma50_available else None,
                    'RSI': hist['RSI'][-1] if 'RSI' in hist else None,
                    'Market Cap': get_market_cap(stock),
                    'Days Data': len(hist)
                })
                
        except Exception as e:
            debug_reasons[str(e)].append(symbol)
            log_debug(symbol, f"Error: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Display debug information
    if debug_mode and debug_reasons:
        st.subheader("Breakout Scan Debug: Rejection Reasons")
        debug_df = pd.DataFrame([
            {"Reason": reason, "Count": len(syms), "Sample": ", ".join(syms[:3])}
            for reason, syms in debug_reasons.items() if len(syms) > 0
        ]).sort_values("Count", ascending=False)
        
        st.dataframe(debug_df)
        
        # Log summary
        logging.info(f"Breakout Scan Summary ({total_symbols} symbols scanned):")
        logging.info(f"Found {len(results)} valid breakouts")
        for reason, count in sorted(debug_reasons.items(), key=lambda x: len(x[1]), reverse=True):
            logging.info(f"{reason}: {len(count)} symbols")
    
    return pd.DataFrame(results)

# [Rest of your functions (display_breakouts, main, etc.) remain the same]

def main():
    try:
        # [Previous sidebar controls setup...]
        
        # Add a note about data requirements
        st.sidebar.markdown("""
        **Note:** Breakout scan requires at least 3 months of historical data
        for accurate SMA calculations. Stocks with insufficient data will be skipped.
        """)
        
        # [Rest of your main function...]
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.error(f"Main error: {str(e)}")

if __name__ == "__main__":
    main()
