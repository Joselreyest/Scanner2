import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from io import StringIO

# Initialize technical indicators (simplified version)
class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df):
        if len(df) >= 20:
            df['SMA20'] = df['Close'].rolling(20).mean()
        if len(df) >= 50:
            df['SMA50'] = df['Close'].rolling(50).mean()
        return df

tech = TechnicalIndicators()

# Set page config
st.set_page_config(
    page_title="Stock Scanner Pro",
    page_icon="📈",
    layout="wide"
)

def display_premarket(df):
    """Display premarket movers"""
    if df.empty:
        st.warning("No premarket gappers found matching your criteria")
        return
    
    st.subheader("Pre-Market Gappers")
    
    # Try to format the numeric columns
    format_columns = {
        'Close': '${:.2f}',
        '% Change': '{:.2f}%',
        'Volume': '{:,}'
    }
    
    # Apply formatting to existing columns
    for col, fmt in format_columns.items():
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

# [Rest of your functions (get_unusual_volume, get_breakouts, get_sp500_symbols) remain the same...]

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
