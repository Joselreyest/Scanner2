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
import pytz
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scanner_debug.log')
    ]
)

# Initialize debug statistics
debug_stats = defaultdict(int)

# Email configuration
if 'email_config' not in st.session_state:
    st.session_state.email_config = {
        'smtp_server': '',
        'smtp_port': 587,
        'email_from': '',
        'email_password': '',
        'email_to': ''
    }

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
    page_title="Advanced Stock Scanner Pro",
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
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return table[0]['Symbol'].tolist()
        except Exception as e:
            log_debug("SYMBOL_FETCH", f"Failed to get SP500 from Wikipedia: {str(e)}")
            return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM']

@st.cache_data(ttl=86400)
def get_nasdaq_symbols():
    """Get Nasdaq-listed symbols using more reliable method"""
    try:
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
    
    etfs = ['IWM', 'VB', 'SCHA', 'FNDA', 'SMLF']
    for etf in etfs:
        try:
            stock = yf.Ticker(etf)
            holdings = stock.get_holdings()
            if holdings is not None and not holdings.empty:
                small_caps += holdings.index.tolist()
        except Exception as e:
            log_debug("SMALL_CAP", f"Failed to get holdings for {etf}: {str(e)}")
    
    known_small_caps = [
        'PLUG', 'FCEL', 'BLNK', 'SNDL', 'NIO', 'XPEV', 'LI', 'WKHS',
        'MVIS', 'SPCE', 'RIDE', 'NKLA', 'BNGO', 'CTRM', 'SENS', 'FUV',
        'GOEV', 'ARVL', 'LCID', 'RIVN', 'PSFE', 'CLOV', 'WISH', 'BBIG',
        'ATER', 'PROG', 'CEI', 'ANY', 'SDC', 'BKKT', 'ATER', 'MMAT'
    ]
    small_caps += known_small_caps
    
    small_caps = list(set([s for s in small_caps if isinstance(s, str) and 1 < len(s) < 6]))
    
    log_debug("SMALL_CAP", f"Loaded {len(small_caps)} small cap symbols")
    return small_caps[:10000]  # Increased limit to 10,000

def get_custom_symbols(uploaded_file):
    """Get symbols from uploaded CSV file"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Symbol' in df.columns:
                return df['Symbol'].tolist()
            elif 'symbol' in df.columns:
                return df['symbol'].tolist()
            else:
                return df.iloc[:, 0].tolist()
        return []
    except Exception as e:
        log_debug("CUSTOM_SYMBOLS", f"Error reading custom symbols: {str(e)}")
        return []

def get_symbols_to_scan():
    """Get combined list of symbols based on user selection"""
    selected_exchanges = st.session_state.get('exchanges', ['SP500'])
    custom_symbols = st.session_state.get('custom_symbols', [])
    
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
    
    if 'CUSTOM' in selected_exchanges and custom_symbols:
        symbols += custom_symbols
        log_debug("SYMBOLS", f"Loaded {len(custom_symbols)} custom symbols")
    
    symbols = list(set(symbols))
    np.random.shuffle(symbols)
    log_debug("SYMBOLS", f"Total unique symbols to scan: {len(symbols)}")
    return symbols

def is_premarket_hours():
    """Check if current time is within pre-market hours (4:00 AM - 9:30 AM ET)"""
    now = datetime.now().astimezone(pytz.timezone('US/Eastern'))
    premarket_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
    premarket_end = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return premarket_start <= now < premarket_end

def get_premarket_movers(min_price, min_volume):
    """Get pre-market movers with improved reliability"""
    try:
        sources = [
            get_benzinga_premarket(),
            get_yahoo_premarket(),
            get_marketwatch_premarket()
        ]
        
        combined_df = pd.concat([df for df in sources if df is not None and not df.empty])
        
        if combined_df.empty:
            log_debug("PRE-MARKET", "All pre-market sources returned empty data")
            return pd.DataFrame()
        
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
        
        for old_col, new_col in column_mapping.items():
            if old_col in combined_df.columns:
                combined_df.rename(columns={old_col: new_col}, inplace=True)
        
        numeric_cols = {'price', 'change', 'volume'}
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(
                    combined_df[col].astype(str).str.replace('[\$,%]', '', regex=True),
                    errors='coerce'
                )
        
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
    with st.expander(f"🔍 {scan_type} Scan Debug Info"):
        if debug_reasons:
            debug_df = pd.DataFrame([
                {"Reason": reason, "Count": len(symbols), "Sample": ", ".join(symbols[:3])}
                for reason, symbols in debug_reasons.items()
            ]).sort_values("Count", ascending=False)
            
            st.dataframe(debug_df)
            st.write(f"Total symbols scanned: {total_scanned}")
            st.write(f"Rejected: {sum(len(v) for v in debug_reasons.values())}")
            
            try:
                with open('scanner_debug.log', 'r') as f:
                    st.subheader("Full Debug Log")
                    st.text(f.read())
            except Exception as e:
                st.warning(f"Could not read debug log file: {str(e)}")
        else:
            st.write("No debug information available")

def plot_stock_chart(symbol, period="1mo"):
    """Plot stock chart for the given symbol"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            st.warning(f"No data available for {symbol}")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist.index, hist['Close'], label='Close Price', color='blue')
        
        if 'SMA20' in hist.columns:
            ax.plot(hist.index, hist['SMA20'], label='20-day SMA', color='orange', linestyle='--')
        if 'SMA50' in hist.columns:
            ax.plot(hist.index, hist['SMA50'], label='50-day SMA', color='green', linestyle='--')
        
        ax.set_title(f"{symbol} Price Chart ({period})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.grid(True)
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error plotting chart for {symbol}: {str(e)}")

def send_email(subject, body):
    """Send email with scan results"""
    try:
        config = st.session_state.email_config
        if not all(config.values()):
            st.error("Email configuration incomplete")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = config['email_from']
        msg['To'] = config['email_to']
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['email_from'], config['email_password'])
        text = msg.as_string()
        server.sendmail(config['email_from'], config['email_to'], text)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def run_scheduled_scan():
    """Run scheduled scan and send email if configured"""
    if 'scan_params' not in st.session_state:
        return
    
    params = st.session_state.scan_params
    symbols = get_symbols_to_scan()
    
    if params['scan_type'] == "Pre-Market Gappers":
        results = get_premarket_movers(params['min_price'], params['min_volume'])
    elif params['scan_type'] == "Unusual Volume":
        results = get_unusual_volume(params['min_price'], params['min_volume'], symbols, params['max_symbols'])
    else:
        results = get_breakouts(params['min_price'], params['min_volume'], symbols, params['max_symbols'])
    
    if not results.empty and st.session_state.email_config['email_to']:
        subject = f"Stock Scanner Results - {params['scan_type']}"
        body = results.to_string()
        send_email(subject, body)

def start_scheduler(interval_minutes):
    """Start background scheduler"""
    schedule.every(interval_minutes).minutes.do(run_scheduled_scan)
    
    def run_continuously():
        while st.session_state.scheduler_running:
            schedule.run_pending()
            time.sleep(1)
    
    scheduler_thread = threading.Thread(target=run_continuously)
    scheduler_thread.start()
    st.session_state.scheduler_thread = scheduler_thread

def stop_scheduler():
    """Stop background scheduler"""
    if 'scheduler_running' in st.session_state:
        st.session_state.scheduler_running = False
    if 'scheduler_thread' in st.session_state:
        st.session_state.scheduler_thread.join()

def display_results_with_chart(df, scan_type):
    """Display results with chart selector that persists"""
    if df.empty:
        st.warning(f"No {scan_type} stocks found")
        return
    
    st.session_state.scan_results = df
    st.session_state.scan_type = scan_type
    
    st.subheader(f"{scan_type} Results")
    
    if scan_type == "Pre-Market Gappers":
        format_dict = {
            'price': '${:.2f}',
            'change': '{:.2f}%',
            'volume': '{:,}'
        }
        symbol_col = 'symbol'
    else:
        format_dict = {
            'Price': '${:.2f}',
            '% Change': '{:.2f}%',
            'Volume': '{:,}',
            'Avg Volume': '{:,}',
            'Volume Ratio': '{:.2f}x',
            'SMA20': '{:.2f}',
            'SMA50': '{:.2f}',
            'RSI': '{:.2f}'
        }
        symbol_col = 'Symbol'
    
    formatted_df = df.copy()
    for col, fmt in format_dict.items():
        if col in formatted_df.columns:
            try:
                formatted_df[col] = formatted_df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else x)
            except:
                pass
    
    st.dataframe(formatted_df, use_container_width=True)
    
    if not df.empty:
        selected_symbol = st.selectbox(
            "Select a symbol to view chart:",
            options=df[symbol_col].unique(),
            key=f"chart_select_{scan_type}_{time.time()}"  # Unique key
        )
        
        if selected_symbol:
            plot_stock_chart(selected_symbol)

def main():
    try:
        st.sidebar.header("Scanner Settings")
        
        # Exchange selection
        st.session_state.exchanges = st.sidebar.multiselect(
            "Exchanges to Scan",
            ['SP500', 'NASDAQ', 'SMALLCAP', 'CUSTOM'],
            default=['SP500', 'NASDAQ']
        )
        
        # Custom symbols upload
        if 'CUSTOM' in st.session_state.exchanges:
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV with Symbols", 
                type=['csv'],
                help="Upload a CSV file with a 'Symbol' column or a single column of symbols"
            )
            st.session_state.custom_symbols = get_custom_symbols(uploaded_file)
        
        scan_type = st.sidebar.selectbox(
            "Scan Type",
            ["Pre-Market Gappers", "Unusual Volume", "Breakouts"]
        )
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_price = st.slider(
                "Min Price", 
                min_value=0.1, 
                max_value=1000.0, 
                value=2.0,
                step=0.1
            )
        with col2:
            min_volume = st.slider(
                "Min Volume (K)", 
                min_value=10, 
                max_value=10000, 
                value=100,
                step=10
            )
        min_volume *= 1000
        
        max_symbols_to_scan = st.sidebar.slider(
            "Max Symbols to Scan",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        # Email configuration
        with st.sidebar.expander("Email Configuration"):
            st.session_state.email_config['smtp_server'] = st.text_input(
                "SMTP Server",
                value=st.session_state.email_config['smtp_server']
            )
            st.session_state.email_config['smtp_port'] = st.number_input(
                "SMTP Port",
                value=st.session_state.email_config['smtp_port'],
                min_value=1,
                max_value=65535
            )
            st.session_state.email_config['email_from'] = st.text_input(
                "From Email",
                value=st.session_state.email_config['email_from']
            )
            st.session_state.email_config['email_password'] = st.text_input(
                "Email Password",
                type="password",
                value=st.session_state.email_config['email_password']
            )
            st.session_state.email_config['email_to'] = st.text_input(
                "To Email",
                value=st.session_state.email_config['email_to']
            )
        
        # Scheduler configuration
        with st.sidebar.expander("Scheduled Scans"):
            schedule_enabled = st.checkbox("Enable Scheduled Scans")
            interval_minutes = st.number_input(
                "Scan Interval (minutes)",
                min_value=1,
                max_value=1440,
                value=60
            )
            
            if schedule_enabled:
                if st.button("Start Scheduled Scans"):
                    st.session_state.scan_params = {
                        'scan_type': scan_type,
                        'min_price': min_price,
                        'min_volume': min_volume,
                        'max_symbols': max_symbols_to_scan
                    }
                    st.session_state.scheduler_running = True
                    start_scheduler(interval_minutes)
                    st.success(f"Scheduled scans started every {interval_minutes} minutes")
            else:
                if st.button("Stop Scheduled Scans"):
                    stop_scheduler()
                    st.success("Scheduled scans stopped")
        
        st.title("📈 Advanced Stock Scanner Pro")
        
        if st.sidebar.button("Run Scan Now"):
            with st.spinner("Loading symbols and scanning..."):
                symbols = get_symbols_to_scan()
                st.write(f"Loaded {len(symbols)} symbols from selected exchanges")
                st.write(f"Scanning first {max_symbols_to_scan} symbols...")
                
                if scan_type == "Pre-Market Gappers":
                    results = get_premarket_movers(min_price, min_volume)
                elif scan_type == "Unusual Volume":
                    results = get_unusual_volume(min_price, min_volume, symbols, max_symbols_to_scan)
                else:
                    results = get_breakouts(min_price, min_volume, symbols, max_symbols_to_scan)
                
                display_results_with_chart(results, scan_type)
                
                if not results.empty and st.session_state.email_config['email_to']:
                    if st.button("Email Results"):
                        subject = f"Stock Scanner Results - {scan_type}"
                        body = results.to_string()
                        if send_email(subject, body):
                            st.success("Results emailed successfully")
        
        if 'scan_results' in st.session_state and 'scan_type' in st.session_state:
            st.write("---")
            display_results_with_chart(st.session_state.scan_results, st.session_state.scan_type)
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.error(f"Main error: {str(e)}")

if __name__ == "__main__":
    main()
