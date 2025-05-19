import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from io import StringIO

# Initialize technical indicators
class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df):
        if len(df) >= 20:
            df['SMA20'] = df['Close'].rolling(20).mean()
        if len(df) >= 50:
            df['SMA50'] = df['Close'].rolling(50).mean()
        
        # Add RSI
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

tech = TechnicalIndicators()

# Set page config
st.set_page_config(
    page_title="Stock Scanner Pro",
    page_icon="📈",
    layout="wide"
)

def get_sp500_symbols():
    """Get current S&P 500 symbols"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return table[0]['Symbol'].tolist()
    except:
        # Fallback if Wikipedia fails
        return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM']

def get_premarket_movers(min_price, min_volume):
    """Get pre-market movers with filters"""
    try:
        url = "https://www.benzinga.com/premarket"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if table is None:
            st.warning("Could not find premarket data table")
            return pd.DataFrame()
        
        # Parse the HTML table
        try:
            df = pd.read_html(StringIO(str(table)))[0]
        except Exception as e:
            st.warning(f"Error parsing table: {e}")
            return pd.DataFrame()
        
        # Clean up column names
        df.columns = [col.strip() for col in df.columns]
        
        # Handle different column name patterns
        column_mapping = {
            'Ticker': 'Symbol',
            'Close▲▼': 'Close',
            '±%': '% Change',
            'Avg. Vol▲▼': 'Volume'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Convert numeric columns
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'].replace('[\$,]', '', regex=True), errors='coerce')
        if '% Change' in df.columns:
            df['% Change'] = pd.to_numeric(df['% Change'].replace('[\%]', '', regex=True), errors='coerce')
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'].replace('[\$,]', '', regex=True), errors='coerce')
        
        # Filter and return
        if 'Close' in df.columns and 'Volume' in df.columns:
            return df[
                (df['Close'] > min_price) & 
                (df['Volume'] > min_volume)
            ].copy()
        else:
            st.warning(f"Missing required columns. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching premarket data: {str(e)}")
        return pd.DataFrame()

def get_unusual_volume(min_price, min_volume):
    """Scan for unusual volume stocks"""
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
                    'RSI': hist['RSI'][-1] if 'RSI' in hist else None
                })
        except Exception as e:
            st.warning(f"Error scanning {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

def display_premarket(df):
    """Display premarket movers"""
    if df.empty:
        st.warning("No premarket gappers found matching your criteria")
        return
    
    st.subheader("Pre-Market Gappers")
    
    # Format numeric columns
    format_dict = {
        'Close': '${:.2f}',
        '% Change': '{:.2f}%',
        'Volume': '{:,}'
    }
    
    for col, fmt in format_dict.items():
        if col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else x)
            except:
                pass
    
    st.dataframe(df, use_container_width=True)

def display_unusual_volume(df):
    """Display unusual volume stocks"""
    if df.empty:
        st.warning("No unusual volume stocks found")
        return
    
    st.subheader("Unusual Volume Stocks")
    
    # Format numeric columns
    if not df.empty:
        df = df.sort_values('Volume Ratio', ascending=False)
        format_dict = {
            'Price': '${:.2f}',
            '% Change': '{:.2f}%',
            'Volume': '{:,}',
            'Avg Volume': '{:,}',
            'Volume Ratio': '{:.2f}x'
        }
        
        for col, fmt in format_dict.items():
            if col in df.columns:
                try:
                    df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else x)
                except:
                    pass
    
    st.dataframe(df, use_container_width=True)

def display_breakouts(df):
    """Display breakout stocks"""
    if df.empty:
        st.warning("No breakout stocks found")
        return
    
    st.subheader("Breakout Candidates")
    
    # Format numeric columns
    if not df.empty:
        format_dict = {
            'Price': '${:.2f}',
            'Breakout Level': '${:.2f}',
            'Volume': '{:,}',
            'SMA20': '{:.2f}',
            'SMA50': '{:.2f}',
            'RSI': '{:.2f}'
        }
        
        for col, fmt in format_dict.items():
            if col in df.columns:
                try:
                    df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else x)
                except:
                    pass
    
    st.dataframe(df, use_container_width=True)

def main():
    try:
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
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
