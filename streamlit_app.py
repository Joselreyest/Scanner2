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

# Initialize technical indicators
class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df, min_periods=20):
        """Add technical indicators with proper period checks"""
        try:
            if len(df) < min_periods:
                return df
            
            if len(df) >= 20:
                df['SMA20'] = df['Close'].rolling(window=20, min_periods=20).mean()
            if len(df) >= 50:
                df['SMA50'] = df['Close'].rolling(window=50, min_periods=50).mean()
            
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
    page_title="Stock Scanner Pro",
    page_icon="📈",
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
    """Get current S&P 500 symbols from reliable source"""
    try:
        # Alternative reliable source for S&P 500
        url = "https://www.slickcharts.com/sp500"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        df = pd.read_html(str(table))[0]
        return df['Symbol'].tolist()
    except Exception as e:
        log_debug("SYMBOL_FETCH", f"Failed to get SP500 from SlickCharts: {str(e)}")
        try:
            # Fallback to Wikipedia
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return table[0]['Symbol'].tolist()
        except Exception as e:
            log_debug("SYMBOL_FETCH", f"Failed to get SP500 from Wikipedia: {str(e)}")
            return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM']

@st.cache_data(ttl=86400)
def get_nasdaq_symbols():
    """Get Nasdaq-listed symbols using more reliable method"""
    try:
        # Use NASDAQ API directly
        url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        return [item['symbol'] for item in data['data']['table']['rows']]
    except Exception as e:
        log_debug("SYMBOL_FETCH", f"Failed to get NASDAQ from API: {str(e)}")
        try:
            # Fallback to CSV download
            url = "https://www.nasdaq.com/market-activity/stocks/screener"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            download_link = soup.find('a', {'data-test': 'download-table'})['href']
            df = pd.read_csv(download_link)
            return df['Symbol'].tolist()
        except Exception as e:
            log_debug("SYMBOL_FETCH", f"Failed to get NASDAQ from CSV: {str(e)}")
            return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM']

@st.cache_data(ttl=86400)
def get_small_cap_symbols():
    """Get small cap stock symbols from reliable sources"""
    small_caps = []
    
    # Source 1: Known small cap ETFs
    etfs = ['IWM', 'VB', 'SCHA', 'FNDA', 'SMLF']
    for etf in etfs:
        try:
            stock = yf.Ticker(etf)
            holdings = stock.get_holdings()
            if holdings is not None and not holdings.empty:
                small_caps += holdings.index.tolist()
        except Exception as e:
            log_debug("SMALL_CAP", f"Failed to get holdings for {etf}: {str(e)}")
    
    # Source 2: Known small cap stocks
    known_small_caps = [
        'PLUG', 'FCEL', 'BLNK', 'SNDL', 'NIO', 'XPEV', 'LI', 'WKHS',
        'MVIS', 'SPCE', 'RIDE', 'NKLA', 'BNGO', 'CTRM', 'SENS', 'FUV',
        'GOEV', 'ARVL', 'LCID', 'RIVN', 'PSFE', 'CLOV', 'WISH', 'BBIG',
        'ATER', 'PROG', 'CEI', 'ANY', 'SDC', 'BKKT', 'ATER', 'MMAT'
    ]
    small_caps += known_small_caps
    
    # Remove duplicates and invalid symbols
    small_caps = list(set([s for s in small_caps if isinstance(s, str) and 1 < len(s) < 6]))
    
    log_debug("SMALL_CAP", f"Loaded {len(small_caps)} small cap symbols")
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
    
    # Remove duplicates and shuffle
    symbols = list(set(symbols))
    np.random.shuffle(symbols)
    log_debug("SYMBOLS", f"Total unique symbols to scan: {len(symbols)}")
    return symbols

def get_premarket_movers(min_price, min_volume):
    """Get pre-market movers with improved reliability"""
    try:
        # Try multiple pre-market data sources
        sources = [
            get_benzinga_premarket(),
            get_yahoo_premarket(),
            get_marketwatch_premarket()
        ]
        
        # Combine all sources and clean the data
        combined_df = pd.concat([df for df in sources if df is not None and not df.empty])
        
        if combined_df.empty:
            log_debug("PRE-MARKET", "All pre-market sources returned empty data")
            return pd.DataFrame()
        
        # Standardize column names
        combined_df.columns = combined_df.columns.str.lower()
        column_mapping = {
            'ticker': 'symbol',
            'last sale': 'price',
            'price': 'price',
            'close': 'price',
            '% change': 'change',
            'change': 'change',
            'volume': 'volume',
            'avg vol': 'volume'
        }
        
        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in combined_df.columns:
                combined_df.rename(columns={old_col: new_col}, inplace=True)
        
        # Convert numeric columns
        numeric_cols = {'price', 'change', 'volume'}
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(
                    combined_df[col].astype(str).str.replace('[\$,%]', '', regex=True),
                    errors='coerce'
                )
        
        # Calculate percentage change if not already available
        if 'change' not in combined_df.columns and 'price' in combined_df.columns:
            # This would require additional logic to calculate changes
            pass
        
        # Filter based on criteria
        required_cols = {'symbol', 'price', 'change', 'volume'}
        if not required_cols.issubset(combined_df.columns):
            missing = required_cols - set(combined_df.columns)
            log_debug("PRE-MARKET", f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        filtered = combined_df[
            (combined_df['price'] > min_price) & 
            (combined_df['volume'] > min_volume)
        ].copy()
        
        if filtered.empty:
            sample = combined_df.sample(min(3, len(combined_df)))
            log_debug("PRE-MARKET", "Sample rejected symbols:")
            for _, row in sample.iterrows():
                log_debug("PRE-MARKET", 
                    f"Symbol: {row['symbol']}, Price: {row.get('price', 'N/A')}, "
                    f"Volume: {row.get('volume', 'N/A')}, "
                    f"Change: {row.get('change', 'N/A')}%"
                )
        
        return filtered
    
    except Exception as e:
        log_debug("PRE-MARKET", f"Error in get_premarket_movers: {str(e)}")
        return pd.DataFrame()

def get_benzinga_premarket():
    """Get pre-market data from Benzinga"""
    try:
        url = "https://www.benzinga.com/premarket"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if table is None:
            log_debug("BENZINGA", "Could not find premarket data table")
            return pd.DataFrame()
        
        df = pd.read_html(StringIO(str(table)))[0]
        log_debug("BENZINGA", f"Found {len(df)} premarket movers")
        return df
    except Exception as e:
        log_debug("BENZINGA", f"Error: {str(e)}")
        return pd.DataFrame()

def get_yahoo_premarket():
    """Get pre-market data from Yahoo Finance"""
    try:
        url = "https://finance.yahoo.com/pre-market"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if table is None:
            log_debug("YAHOO", "Could not find premarket data table")
            return pd.DataFrame()
        
        df = pd.read_html(StringIO(str(table)))[0]
        log_debug("YAHOO", f"Found {len(df)} premarket movers")
        return df
    except Exception as e:
        log_debug("YAHOO", f"Error: {str(e)}")
        return pd.DataFrame()

def get_marketwatch_premarket():
    """Get pre-market data from MarketWatch"""
    try:
        url = "https://www.marketwatch.com/tools/screener/premarket"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if table is None:
            log_debug("MARKETWATCH", "Could not find premarket data table")
            return pd.DataFrame()
        
        df = pd.read_html(StringIO(str(table)))[0]
        log_debug("MARKETWATCH", f"Found {len(df)} premarket movers")
        return df
    except Exception as e:
        log_debug("MARKETWATCH", f"Error: {str(e)}")
        return pd.DataFrame()

def display_premarket(df):
    """Enhanced pre-market display with more information"""
    if df.empty:
        st.warning("""
        No premarket gappers found matching your criteria. This could be because:
        - It's outside pre-market hours (4:00 AM - 9:30 AM ET)
        - Your price/volume filters are too restrictive
        - Data sources are temporarily unavailable
        """)
        
        if debug_mode:
            st.info("Try lowering your minimum price/volume requirements or check debug logs")
        return
    
    st.subheader("Pre-Market Gappers")
    
    # Add additional metrics
    if 'change' in df.columns:
        df['change'] = pd.to_numeric(df['change'], errors='coerce')
        df = df.sort_values('change', ascending=False)
    
    # Format display
    format_dict = {
        'price': '${:.2f}',
        'change': '{:.2f}%',
        'volume': '{:,}'
    }
    
    for col, fmt in format_dict.items():
        if col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else 'N/A')
            except:
                pass
    
    # Display with additional info
    st.dataframe(
        df[['symbol', 'price', 'change', 'volume']],
        use_container_width=True,
        height=min(800, 35 * (len(df) + 1))
    
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def is_premarket_hours():
    now = datetime.now().astimezone(pytz.timezone('US/Eastern'))
    premarket_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
    premarket_end = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return premarket_start <= now < premarket_end


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
        **Data Sources:**  
        - SP500: SlickCharts + Wikipedia  
        - NASDAQ: Official API + CSV  
        - Small Caps: ETF Holdings + Known Stocks
        """)
        
        st.title("📈 Robust Stock Scanner Pro")
        
        if st.sidebar.button("Run Scan"):
            with st.spinner("Loading symbols and scanning..."):
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
