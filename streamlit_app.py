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

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="Stock Scanner Pro",
    page_icon="📈",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Advanced Stock Scanner with Breakout Detection"
    }
)

# Debug toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

def log_debug(symbol, message):
    """Log debug messages if debug mode is enabled"""
    debug_stats[message] += 1
    if debug_mode:
        logging.info(f"{symbol}: {message}")

@st.cache_data(ttl=86400)
def get_sp500_symbols():
    """Get current S&P 500 symbols"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return table[0]['Symbol'].tolist()
    except Exception as e:
        log_debug("SYMBOL_FETCH", f"Failed to get SP500 symbols: {str(e)}")
        return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM']

@st.cache_data(ttl=86400)
def get_nasdaq_symbols():
    """Get Nasdaq-listed symbols from a reliable source"""
    try:
        nasdaq_url = "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(nasdaq_url, headers=headers, timeout=15)
        df = pd.read_csv(StringIO(response.text))
        return df['Symbol'].tolist()
    except Exception as e:
        log_debug("SYMBOL_FETCH", f"Failed to get NASDAQ symbols: {str(e)}")
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
            return table['Ticker'].tolist()
        except:
            return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM']

@st.cache_data(ttl=86400)
def get_small_cap_symbols():
    """Get small cap stock symbols from multiple reliable sources"""
    small_caps = []
    try:
        # Try to get Russell 2000 components
        russell_url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax"
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {
            "fileType": "csv",
            "fileName": "IWM_holdings",
            "dataType": "fund"
        }
        response = requests.get(russell_url, headers=headers, params=params, timeout=15)
        df = pd.read_csv(StringIO(response.text))
        small_caps += df['Ticker'].dropna().tolist()
    except Exception as e:
        log_debug("SMALL_CAP", f"Failed to get Russell 2000: {str(e)}")
    
    # Add known small caps
    small_caps += ['PLUG', 'FCEL', 'BLNK', 'SNDL', 'NIO', 'XPEV', 'LI', 'WKHS', 
                  'MVIS', 'SPCE', 'RIDE', 'NKLA', 'BNGO', 'CTRM', 'SENS']
    
    return list(set(small_caps))[:1000]  # Remove duplicates and limit

def get_symbols_to_scan():
    """Get combined list of symbols based on user selection"""
    selected_exchanges = st.session_state.get('exchanges', ['SP500'])
    
    symbols = []
    if 'SP500' in selected_exchanges:
        symbols += get_sp500_symbols()
    if 'NASDAQ' in selected_exchanges:
        symbols += get_nasdaq_symbols()
    if 'SMALLCAP' in selected_exchanges:
        symbols += get_small_cap_symbols()
    
    # Remove duplicates and shuffle
    symbols = list(set(symbols))
    np.random.shuffle(symbols)
    return symbols

def get_premarket_movers(min_price, min_volume):
    """Get pre-market movers with filters"""
    try:
        url = "https://www.benzinga.com/premarket"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if table is None:
            log_debug("PRE-MARKET", "Could not find premarket data table")
            return pd.DataFrame()
        
        try:
            df = pd.read_html(StringIO(str(table)))[0]
        except Exception as e:
            log_debug("PRE-MARKET", f"Error parsing table: {e}")
            return pd.DataFrame()
        
        # Clean and filter data
        df.columns = [col.strip() for col in df.columns]
        column_mapping = {
            'Ticker': 'Symbol',
            'Close▲▼': 'Close',
            '±%': '% Change',
            'Avg. Vol▲▼': 'Volume',
            'Company': 'Symbol',
            'Price': 'Close'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Convert numeric columns
        numeric_cols = {'Close', '% Change', 'Volume'}
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace('[\$,%]', '', regex=True), errors='coerce')
        
        # Filter and return
        if 'Close' in df.columns and 'Volume' in df.columns:
            return df[(df['Close'] > min_price) & (df['Volume'] > min_volume)].copy()
        return pd.DataFrame()
            
    except Exception as e:
        log_debug("PRE-MARKET", f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def get_unusual_volume(min_price, min_volume, symbols, max_symbols_to_scan):
    """Scan for unusual volume stocks"""
    results = []
    debug_reasons = defaultdict(list)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    symbols_to_scan = symbols[:max_symbols_to_scan]
    
    for i, symbol in enumerate(symbols_to_scan):
        try:
            status_text.text(f"Scanning {symbol} ({i+1}/{len(symbols_to_scan)})")
            progress_bar.progress((i+1)/len(symbols_to_scan))
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d", interval="5m")
            
            if len(hist) < 20 or hist.empty:
                debug_reasons["Insufficient data"].append(symbol)
                continue
                
            current_volume = hist['Volume'][-1]
            avg_volume = hist['Volume'].mean()
            current_close = hist['Close'][-1]
            prev_close = hist['Close'][-2] if len(hist) > 1 else hist['Open'][0]
            price_change_pct = (current_close - prev_close) / prev_close * 100
            
            # Check conditions
            conditions = [
                (current_close > min_price, "Price too low"),
                (current_volume > min_volume, "Volume too low"),
                (current_volume > 2 * avg_volume, "Volume ratio too low"),
                (price_change_pct > 1.0, "Price change too small")
            ]
            
            for condition, reason in conditions:
                if not condition:
                    debug_reasons[reason].append(symbol)
                    break
            else:
                results.append({
                    'Symbol': symbol,
                    'Price': current_close,
                    '% Change': price_change_pct,
                    'Volume': current_volume,
                    'Avg Volume': avg_volume,
                    'Volume Ratio': current_volume / avg_volume,
                    'Market Cap': get_market_cap(stock)
                })
                
        except Exception as e:
            debug_reasons[str(e)].append(symbol)
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if debug_mode:
        display_debug_info("Unusual Volume", debug_reasons, len(symbols_to_scan))
    
    return pd.DataFrame(results)

def get_breakouts(min_price, min_volume, symbols, max_symbols_to_scan):
    """Scan for breakout stocks with proper SMA handling"""
    results = []
    debug_reasons = defaultdict(list)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    symbols_to_scan = symbols[:max_symbols_to_scan]
    
    for i, symbol in enumerate(symbols_to_scan):
        try:
            status_text.text(f"Scanning {symbol} ({i+1}/{len(symbols_to_scan)})")
            progress_bar.progress((i+1)/len(symbols_to_scan))
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period="3mo")
            
            if len(hist) < 50:
                debug_reasons["Insufficient history (<50 days)"].append(symbol)
                continue
                
            hist = tech.add_all_indicators(hist)
            
            current_price = hist['Close'][-1]
            current_volume = hist['Volume'][-1]
            recent_high = hist['High'][-5:-1].max()
            
            # Check conditions
            conditions = [
                (current_price > min_price, "Price too low"),
                (current_volume > min_volume, "Volume too low"),
                (current_price > recent_high, "Not breaking out"),
                ('SMA20' in hist, "SMA20 not available"),
                ('SMA50' in hist, "SMA50 not available"),
                (not ('SMA20' in hist and 'SMA50' in hist) or hist['SMA20'][-1] > hist['SMA50'][-1], 
                 "SMA crossover invalid")
            ]
            
            for condition, reason in conditions:
                if not condition:
                    debug_reasons[reason].append(symbol)
                    break
            else:
                results.append({
                    'Symbol': symbol,
                    'Price': current_price,
                    'Breakout Level': recent_high,
                    'Volume': current_volume,
                    'SMA20': hist['SMA20'][-1],
                    'SMA50': hist['SMA50'][-1],
                    'RSI': hist['RSI'][-1] if 'RSI' in hist else None,
                    'Market Cap': get_market_cap(stock),
                    'Days Data': len(hist)
                })
                
        except Exception as e:
            debug_reasons[str(e)].append(symbol)
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if debug_mode:
        display_debug_info("Breakout", debug_reasons, len(symbols_to_scan))
    
    return pd.DataFrame(results)

def get_market_cap(stock):
    """Get market cap if available"""
    try:
        info = stock.info
        if 'marketCap' in info:
            cap = info['marketCap']
            if cap > 1e9:
                return f"${cap/1e9:.2f}B"
            elif cap > 1e6:
                return f"${cap/1e6:.2f}M"
            return f"${cap:,.0f}"
    except:
        return "N/A"

def display_debug_info(scan_type, debug_reasons, total_scanned):
    """Display debug information for a scan"""
    st.subheader(f"{scan_type} Scan Debug Info")
    
    if debug_reasons:
        debug_df = pd.DataFrame([
            {"Reason": reason, "Count": len(symbols), "Sample": ", ".join(symbols[:3])}
            for reason, symbols in debug_reasons.items()
        ]).sort_values("Count", ascending=False)
        
        st.dataframe(debug_df)
        st.write(f"Total symbols scanned: {total_scanned}")
        st.write(f"Rejected: {sum(len(v) for v in debug_reasons.values())}")
    else:
        st.write("No debug information available")

def display_premarket(df):
    """Display premarket movers"""
    if df.empty:
        st.warning("No premarket gappers found matching your criteria")
        return
    
    st.subheader("Pre-Market Gappers")
    
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
        st.sidebar.header("Scanner Settings")
        
        # Exchange selection
        st.session_state.exchanges = st.sidebar.multiselect(
            "Exchanges to Scan",
            ['SP500', 'NASDAQ', 'SMALLCAP'],
            default=['SP500', 'NASDAQ', 'SMALLCAP']
        )
        
        scan_type = st.sidebar.selectbox(
            "Scan Type",
            ["Pre-Market Gappers", "Unusual Volume", "Breakouts"]
        )
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_price = st.slider(
                "Min Price", 
                min_value=0.1, 
                max_value=100.0, 
                value=2.0,
                step=0.1
            )
        with col2:
            min_volume = st.slider(
                "Min Volume (K)", 
                min_value=10, 
                max_value=5000, 
                value=100,
                step=10
            )
        min_volume *= 1000
        
        max_symbols_to_scan = st.sidebar.slider(
            "Max Symbols to Scan",
            min_value=50,
            max_value=1000,
            value=200,
            step=50
        )
        
        st.sidebar.markdown("""
        **Breakout Scan Note:**  
        Requires at least 3 months of historical data  
        for accurate SMA calculations.
        """)
        
        st.title("📈 Stock Scanner Pro")
        
        if st.sidebar.button("Run Scan"):
            with st.spinner("Scanning stocks..."):
                symbols = get_symbols_to_scan()
                st.write(f"Loaded {len(symbols)} symbols from selected exchanges")
                st.write(f"Scanning first {max_symbols_to_scan} symbols...")
                
                if scan_type == "Pre-Market Gappers":
                    results = get_premarket_movers(min_price, min_volume)
                    display_premarket(results)
                elif scan_type == "Unusual Volume":
                    results = get_unusual_volume(min_price, min_volume, symbols, max_symbols_to_scan)
                    display_unusual_volume(results)
                else:
                    results = get_breakouts(min_price, min_volume, symbols, max_symbols_to_scan)
                    display_breakouts(results)
                
                if debug_mode:
                    try:
                        with open('scanner_debug.log', 'r') as f:
                            st.subheader("Debug Log")
                            st.text(f.read())
                    except:
                        st.warning("Could not read debug log file")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.error(f"Main error: {str(e)}")

if __name__ == "__main__":
    main()
