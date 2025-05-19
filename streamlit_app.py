import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
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
    page_title="Advanced Stock Scanner Pro",
    page_icon="📊",
    layout="wide"
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
        # Alternative source for NASDAQ symbols
        nasdaq_url = "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(nasdaq_url, headers=headers, timeout=15)
        
        # Read the CSV content
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        return df['Symbol'].tolist()
    except Exception as e:
        log_debug("SYMBOL_FETCH", f"Failed to get NASDAQ symbols: {str(e)}")
        try:
            # Fallback to NASDAQ 100
            table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
            return table['Ticker'].tolist()
        except:
            return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM']

@st.cache_data(ttl=86400)
def get_small_cap_symbols():
    """Get small cap stock symbols from multiple reliable sources"""
    small_caps = []
    
    # Source 1: Russell 2000 components
    try:
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
    
    # Source 2: Small cap ETFs holdings
    etfs = ['IWM', 'VB', 'SCHA', 'FNDA', 'SMLF']
    for etf in etfs:
        try:
            stock = yf.Ticker(etf)
            holdings = stock.get_holdings()
            if holdings is not None and not holdings.empty:
                small_caps += holdings.index.tolist()
        except:
            continue
    
    # Source 3: Known small cap lists
    extra_caps = ['PLUG', 'FCEL', 'BLNK', 'SNDL', 'NIO', 'XPEV', 'LI', 'WKHS', 
                 'MVIS', 'SPCE', 'RIDE', 'NKLA', 'WKHS', 'BNGO', 'CTRM', 'SENS']
    small_caps += extra_caps
    
    # Remove duplicates and invalid symbols
    small_caps = list(set([s for s in small_caps if isinstance(s, str) and 1 < len(s) < 6]))
    
    return small_caps[:1000]  # Limit to 1000 symbols

def get_symbols_to_scan():
    """Get combined list of symbols based on user selection"""
    selected_exchanges = st.session_state.get('exchanges', ['SP500'])
    
    symbols = []
    if 'SP500' in selected_exchanges:
        sp500 = get_sp500_symbols()
        symbols += sp500
        log_debug("SYMBOLS", f"Loaded {len(sp500)} SP500 symbols")
    
    if 'NASDAQ' in selected_exchanges:
        nasdaq = get_nasdaq_symbols()
        symbols += nasdaq
        log_debug("SYMBOLS", f"Loaded {len(nasdaq)} NASDAQ symbols")
    
    if 'SMALLCAP' in selected_exchanges:
        small_caps = get_small_cap_symbols()
        symbols += small_caps
        log_debug("SYMBOLS", f"Loaded {len(small_caps)} small cap symbols")
    
    # Remove duplicates and shuffle for better distribution
    symbols = list(set(symbols))
    np.random.shuffle(symbols)
    log_debug("SYMBOLS", f"Total unique symbols to scan: {len(symbols)}")
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
        
        # Clean up column names
        df.columns = [col.strip() for col in df.columns]
        
        # Handle different column name patterns
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
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'].replace('[\$,]', '', regex=True), errors='coerce')
        if '% Change' in df.columns:
            df['% Change'] = pd.to_numeric(df['% Change'].replace('[\%]', '', regex=True), errors='coerce')
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'].replace('[\$,]', '', regex=True), errors='coerce')
        
        # Filter and return
        if 'Close' in df.columns and 'Volume' in df.columns:
            filtered = df[
                (df['Close'] > min_price) & 
                (df['Volume'] > min_volume)
            ].copy()
            
            # Log filtering stats
            log_debug("PRE-MARKET", f"Original: {len(df)}, Filtered: {len(filtered)}")
            if len(df) > 0 and len(filtered) == 0:
                sample = df.sample(min(3, len(df)))
                for _, row in sample.iterrows():
                    log_debug("PRE-MARKET", f"Sample rejected - Price: {row.get('Close', 'N/A')}, Volume: {row.get('Volume', 'N/A')}")
            
            return filtered
        else:
            log_debug("PRE-MARKET", f"Missing required columns. Available: {df.columns.tolist()}")
            return pd.DataFrame()
            
    except Exception as e:
        log_debug("PRE-MARKET", f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def get_unusual_volume(min_price, min_volume, symbols, max_symbols_to_scan):
    """Scan for unusual volume stocks with detailed debugging"""
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
            hist = stock.history(period="1d", interval="5m")
            
            if len(hist) < 20:
                debug_reasons["Not enough data points"].append(symbol)
                continue
                
            if hist.empty:
                debug_reasons["Empty history"].append(symbol)
                continue
                
            current_volume = hist['Volume'][-1]
            avg_volume = hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume
            
            current_close = hist['Close'][-1]
            prev_close = hist['Close'][-2] if len(hist) > 1 else hist['Open'][0]
            price_change_pct = (current_close - prev_close) / prev_close * 100
            
            # Check each condition separately for debugging
            conditions = {
                'Price too low': current_close <= min_price,
                'Volume too low': current_volume <= min_volume,
                'Volume ratio too low': volume_ratio <= 1.5,
                'Price change too small': price_change_pct <= 0.5
            }
            
            rejected = False
            for reason, condition in conditions.items():
                if condition:
                    debug_reasons[reason].append(symbol)
                    log_debug(symbol, f"Unusual Volume Rejected - {reason}")
                    rejected = True
                    break
            
            if not rejected:
                results.append({
                    'Symbol': symbol,
                    'Price': current_close,
                    '% Change': price_change_pct,
                    'Volume': current_volume,
                    'Avg Volume': avg_volume,
                    'Volume Ratio': volume_ratio,
                    'Market Cap': get_market_cap(stock)
                })
                
        except Exception as e:
            debug_reasons[str(e)].append(symbol)
            log_debug(symbol, f"Error: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Display debug information
    if debug_mode and debug_reasons:
        st.subheader("Debug: Rejection Reasons")
        debug_df = pd.DataFrame([
            {"Reason": reason, "Count": len(symbols), "Sample": ", ".join(symbols[:3])}
            for reason, symbols in debug_reasons.items()
        ])
        st.dataframe(debug_df)
        
        # Log summary
        logging.info(f"Unusual Volume Scan Summary ({total_symbols} symbols scanned):")
        for reason, count in debug_stats.items():
            if "Unusual Volume Rejected" in reason:
                logging.info(f"{reason}: {count}")
    
    return pd.DataFrame(results)

def get_breakouts(min_price, min_volume, symbols, max_symbols_to_scan):
    """Scan for breakout stocks with detailed debugging"""
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
            hist = stock.history(period="1mo")
            hist = tech.add_all_indicators(hist)
            
            if len(hist) < 5:
                debug_reasons["Not enough historical data"].append(symbol)
                continue
                
            if hist.empty:
                debug_reasons["Empty history"].append(symbol)
                continue
                
            current_price = hist['Close'][-1]
            current_volume = hist['Volume'][-1]
            avg_volume = hist['Volume'].mean()
            recent_high = hist['High'][-5:-1].max()
            
            # Check each condition separately
            conditions = {
                'Price too low': current_price <= min_price,
                'Volume too low': current_volume <= min_volume,
                'Not breaking out': current_price <= recent_high,
                'SMA crossover invalid': 'SMA20' in hist and hist['SMA20'][-1] <= hist['SMA50'][-1]
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
                    'SMA20': hist['SMA20'][-1] if 'SMA20' in hist else None,
                    'SMA50': hist['SMA50'][-1] if 'SMA50' in hist else None,
                    'RSI': hist['RSI'][-1] if 'RSI' in hist else None,
                    'Market Cap': get_market_cap(stock)
                })
                
        except Exception as e:
            debug_reasons[str(e)].append(symbol)
            log_debug(symbol, f"Error: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Display debug information
    if debug_mode and debug_reasons:
        st.subheader("Debug: Rejection Reasons")
        debug_df = pd.DataFrame([
            {"Reason": reason, "Count": len(symbols), "Sample": ", ".join(symbols[:3])}
            for reason, symbols in debug_reasons.items()
        ])
        st.dataframe(debug_df)
        
        # Log summary
        logging.info(f"Breakout Scan Summary ({total_symbols} symbols scanned):")
        for reason, count in debug_stats.items():
            if "Breakout Rejected" in reason:
                logging.info(f"{reason}: {count}")
    
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
    return "N/A"

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
        
        # Flexible filters
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
        
        # Symbol limit slider
        max_symbols_to_scan = st.sidebar.slider(
            "Max Symbols to Scan",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Limit how many symbols to scan for performance reasons"
        )
        
        # Main app
        st.title("📈 Advanced Stock Scanner Pro")
        
        if st.sidebar.button("Run Scan"):
            with st.spinner("Scanning stocks..."):
                symbols = get_symbols_to_scan()
                st.write(f"Loaded {len(symbols)} total symbols from selected exchanges")
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
                
                # Show debug log
                if debug_mode:
                    st.subheader("Debug Log")
                    with open('scanner_debug.log', 'r') as f:
                        st.text(f.read())
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.error(f"Main error: {str(e)}")

if __name__ == "__main__":
    main()
