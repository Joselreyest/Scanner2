import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
try:
    from utils.technical_indicators import TechnicalIndicators
except ImportError:
    st.error("Technical indicators module not found. Creating dummy class.")
    class TechnicalIndicators:
        @staticmethod
        def add_all_indicators(df):
            return df

# Initialize technical indicators
tech = TechnicalIndicators()

# Set page config
st.set_page_config(
    page_title="Stock Scanner Pro",
    page_icon="📈",
    layout="wide"
)

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
            
        # Use pandas with lxml if available, fallback to html.parser
        try:
            df = pd.read_html(str(table))[0]
        except:
            # Try alternative parsing if lxml fails
            from io import StringIO
            df = pd.read_html(StringIO(str(table)))[0]
        
        # Clean up column names
        df.columns = [col.strip() for col in df.columns]
        
        # Check for required columns
        required_columns = {'Last Sale', 'Volume'}
        if not required_columns.issubset(df.columns):
            st.warning(f"Missing required columns in data. Found: {df.columns.tolist()}")
            return pd.DataFrame()
            
        # Filter and return
        return df[
            (df['Last Sale'].astype(float) > min_price) & 
            (df['Volume'].astype(int) > min_volume)
        ].copy()
        
    except Exception as e:
        st.error(f"Error fetching premarket data: {str(e)}")
        return pd.DataFrame()

# [Rest of your functions remain the same...]

if __name__ == "__main__":
    main()
