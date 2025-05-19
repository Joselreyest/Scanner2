import streamlit as st
import yfinance as yf
import pandas as pd
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
        df = pd.read_html(str(table))[0]
        
        # Filter and return
        return df[
            (df['Last Sale'] > min_price) & 
            (df['Volume'] > min_volume)
        ]
    except Exception as e:
        st.error(f"Error fetching premarket data: {e}")
        return pd.DataFrame()

def get_unusual_volume(min_price, min_volume):
    """Scan for unusual volume stocks"""
    # Get S&P 500 symbols
    symbols = get_sp500_symbols()
    results = []
    
    for symbol in symbols[:50]:  # Limit for demo
        try:
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
        except:
            continue
    
    return pd.DataFrame(results)

def get_breakouts(min_price, min_volume):
    """Scan for breakout stocks"""
    symbols = get_sp500_symbols()
    results = []
    
    for symbol in symbols[:50]:  # Limit for demo
        try:
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
        except:
            continue
    
    return pd.DataFrame(results)

def display_premarket(df):
    """Display premarket movers"""
    if df.empty:
        st.warning("No premarket gappers found matching your criteria")
        return
    
    st.subheader("Pre-Market Gappers")
    st.dataframe(
        df.style.format({
            'Last Sale': '${:.2f}',
            '% Change': '{:.2f}%',
            'Volume': '{:,}'
        }),
        use_container_width=True
    )

def display_unusual_volume(df):
    """Display unusual volume stocks"""
    if df.empty:
        st.warning("No unusual volume stocks found")
        return
    
    st.subheader("Unusual Volume Stocks")
    st.dataframe(
        df.sort_values('Volume Ratio', ascending=False).style.format({
            'Price': '${:.2f}',
            '% Change': '{:.2f}%',
            'Volume': '{:,}',
            'Avg Volume': '{:,}',
            'Volume Ratio': '{:.2f}x'
        }),
        use_container_width=True
    )

def display_breakouts(df):
    """Display breakout stocks"""
    if df.empty:
        st.warning("No breakout stocks found")
        return
    
    st.subheader("Breakout Candidates")
    st.dataframe(
        df.style.format({
            'Price': '${:.2f}',
            'Breakout Level': '${:.2f}',
            'Volume': '{:,}',
            'SMA20': '{:.2f}',
            'SMA50': '{:.2f}',
            'RSI': '{:.2f}'
        }),
        use_container_width=True
    )

def get_sp500_symbols():
    """Get S&P 500 symbols with caching"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return table[0]['Symbol'].tolist()
    except:
        return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM']

if __name__ == "__main__":
    main()
