import streamlit as st
import yfinance as yf
import pandas as pd
import requests  # This was missing
from bs4 import BeautifulSoup
from datetime import datetime
import time
from utils.technical_indicators import TechnicalIndicators

# Initialize technical indicators
tech = TechnicalIndicators()

# Set page config
st.set_page_config(
    page_title="Stock Scanner Pro",
    page_icon="📈",
    layout="wide"
)

def main():
    # Sidebar controls
    st.sidebar.header("Scanner Settings")
    scan_type = st.sidebar.selectbox(
        "Scan Type",
        ["Pre-Market Gappers", "Unusual Volume", "Breakouts"]
    )
    min_price = st.sidebar.number_input("Minimum Price", value=5.0)
    min_volume = st.sidebar.number_input("Minimum Volume (K)", value=500)
    min_volume *= 1000  # Convert to actual volume
    
    # Main app
    st.title("📊 Stock Scanner Pro")
    
    if st.sidebar.button("Run Scan"):
        with st.spinner("Scanning stocks..."):
            # Run the appropriate scan
            if scan_type == "Pre-Market Gappers":
                results = get_premarket_movers(min_price, min_volume)
                display_premarket(results)
            elif scan_type == "Unusual Volume":
                results = get_unusual_volume(min_price, min_volume)
                display_unusual_volume(results)
            else:
                results = get_breakouts(min_price, min_volume)
                display_breakouts(results)

def get_premarket_movers(min_price, min_volume):
    """Get pre-market movers with filters"""
    try:
        url = "https://www.benzinga.com/premarket"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if table is None:
            st.warning("Could not find premarket data table")
            return pd.DataFrame()
            
        df = pd.read_html(str(table))[0]
        
        # Clean up column names (Benzinga sometimes changes these)
        df.columns = [col.strip() for col in df.columns]
        
        # Filter and return
        if 'Last Sale' in df.columns and 'Volume' in df.columns:
            return df[
                (df['Last Sale'] > min_price) & 
                (df['Volume'] > min_volume)
            ].copy()
        else:
            st.warning("Unexpected table format from Benzinga")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching premarket data: {e}")
        return pd.DataFrame()

def get_unusual_volume(min_price, min_volume):
    """Scan for unusual volume stocks"""
    # Get S&P 500 symbols
    symbols = get_sp500_symbols()
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols[:50]):  # Limit for demo
        try:
            status_text.text(f"Scanning {symbol} ({i+1}/50)")
            progress_bar.progress((i+1)/50)
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d", interval="5m")
            
            if len(hist) < 20 or hist.empty:
                continue
                
            current_volume = hist['Volume'][-1]
            avg_volume = hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume
            
            current_close = hist['Close'][-1]
            prev_close = hist['Close'][-2] if len(hist) > 1 else hist['Open'][0]
            price_change_pct = (current_close - prev_close) / prev_close * 100
            
            if (current_close > min_price and 
                current_volume > min_volume and 
                volume_ratio > 2.0 and 
                price_change_pct > 1.0):
                
                results.append({
                    'Symbol': symbol,
                    'Price': current_close,
                    '% Change': price_change_pct,
                    'Volume': current_volume,
                    'Avg Volume': avg_volume,
                    'Volume Ratio': volume_ratio
                })
        except Exception as e:
            st.warning(f"Error scanning {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

def get_breakouts(min_price, min_volume):
    """Scan for breakout stocks"""
    symbols = get_sp500_symbols()
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols[:50]):  # Limit for demo
        try:
            status_text.text(f"Scanning {symbol} ({i+1}/50)")
            progress_bar.progress((i+1)/50)
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo")
            hist = tech.add_all_indicators(hist)
            
            if len(hist) < 5 or hist.empty:
                continue
                
            current_price = hist['Close'][-1]
            current_volume = hist['Volume'][-1]
            avg_volume = hist['Volume'].mean()
            recent_high = hist['High'][-5:-1].max()
            
            if (current_price > min_price and
                current_volume > min_volume and
                current_price > recent_high and
                hist['SMA20'][-1] > hist['SMA50'][-1]):
                
                results.append({
                    'Symbol': symbol,
                    'Price': current_price,
                    'Breakout Level': recent_high,
                    'Volume': current_volume,
                    'SMA20': hist['SMA20'][-1],
                    'SMA50': hist['SMA50'][-1],
                    'RSI': hist['RSI'][-1]
                })
        except Exception as e:
            st.warning(f"Error scanning {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# ... [rest of the file remains the same, including display functions and get_sp500_symbols] ...

if __name__ == "__main__":
    main()
