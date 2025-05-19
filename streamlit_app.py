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

# Email configuration (store in session state)
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

# Email functions
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

# Scheduler functions
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
    
    # Start the scheduler thread
    scheduler_thread = threading.Thread(target=run_continuously)
    scheduler_thread.start()
    st.session_state.scheduler_thread = scheduler_thread

def stop_scheduler():
    """Stop background scheduler"""
    if 'scheduler_running' in st.session_state:
        st.session_state.scheduler_running = False
    if 'scheduler_thread' in st.session_state:
        st.session_state.scheduler_thread.join()

# [Previous functions remain the same until main()]

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
        
        # Email configuration expander
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
                
                # Option to email results
                if not results.empty and st.session_state.email_config['email_to']:
                    if st.button("Email Results"):
                        subject = f"Stock Scanner Results - {scan_type}"
                        body = results.to_string()
                        if send_email(subject, body):
                            st.success("Results emailed successfully")
        
        # Show previous results if they exist
        if 'scan_results' in st.session_state and 'scan_type' in st.session_state:
            st.write("---")
            display_results_with_chart(st.session_state.scan_results, st.session_state.scan_type)
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.error(f"Main error: {str(e)}")

if __name__ == "__main__":
    main()
