import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import schedule
import time
from tqdm import tqdm
from config import Config
from utils.email_alerts import EmailAlerts
from utils.technical_indicators import TechnicalIndicators
from utils.backtesting import Backtester

class StockScanner:
    def __init__(self):
        self.today = datetime.now().date()
        self.unusual_volume_stocks = []
        self.breakout_stocks = []
        self.gappers = []
        self.alert_system = EmailAlerts()
        self.tech_indicators = TechnicalIndicators()
        self.backtester = Backtester()
        
    def get_premarket_movers(self):
        """Get pre-market gainers with volume from multiple sources"""
        try:
            sources = [
                self._get_benzinga_premarket(),
                self._get_yahoo_premarket()
            ]
            
            # Combine and filter results
            all_gappers = pd.concat(sources).drop_duplicates()
            all_gappers = all_gappers[
                (all_gappers['% Change'] > 2) & 
                (all_gappers['% Change'] < 5) &
                (all_gappers['Volume'] > 100000)
            ]
            
            self.gappers = all_gappers.values.tolist()
            
        except Exception as e:
            print(f"Error getting premarket movers: {e}")

    def _get_benzinga_premarket(self):
        """Scrape Benzinga premarket data"""
        url = "https://www.benzinga.com/premarket"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        df = pd.read_html(str(table))[0]
        df.columns = ['Symbol', 'Last Sale', '% Change', 'Volume', 'Time', 'Headlines']
        return df[['Symbol', 'Last Sale', '% Change', 'Volume']]

    def _get_yahoo_premarket(self):
        """Scrape Yahoo Finance premarket data"""
        url = "https://finance.yahoo.com/pre-market"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        df = pd.read_html(str(table))[0]
        df.columns = ['Symbol', 'Name', 'Price', 'Change', '% Change', 'Volume']
        return df[['Symbol', 'Price', '% Change', 'Volume']]

    def scan_unusual_volume(self, symbols):
        """Scan for unusual volume in given symbols"""
        unusual = []
        
        for symbol in tqdm(symbols, desc="Scanning for unusual volume"):
            try:
                # Get recent data - adjust period for pre-market
                period = "1d" if Config.is_pre_market() else "1mo"
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period, interval="5m")
                
                if len(hist) < 20:  # Need enough data points
                    continue
                    
                # Calculate metrics
                current_volume = hist['Volume'][-1]
                avg_volume = hist['Volume'].mean()
                volume_ratio = current_volume / avg_volume
                
                current_close = hist['Close'][-1]
                prev_close = hist['Close'][-2] if len(hist) > 1 else hist['Open'][0]
                price_change_pct = (current_close - prev_close) / prev_close * 100
                
                # Check criteria
                if (volume_ratio > Config.VOLUME_RATIO_THRESHOLD and 
                    price_change_pct > Config.PRICE_CHANGE_THRESHOLD and 
                    current_close > Config.MIN_PRICE and 
                    current_volume > Config.MIN_VOLUME):
                    
                    unusual.append({
                        'symbol': symbol,
                        'price': current_close,
                        'volume': current_volume,
                        'avg_volume': avg_volume,
                        'volume_ratio': round(volume_ratio, 2),
                        'price_change': round(price_change_pct, 2),
                        'time': hist.index[-1]
                    })
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        self.unusual_volume_stocks = unusual

    def scan_breakouts(self, symbols):
        """Scan for potential breakout stocks with technical indicators"""
        breakouts = []
        
        for symbol in tqdm(symbols, desc="Scanning for breakouts"):
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1mo")
                
                if len(hist) < 5:
                    continue
                    
                # Calculate technical indicators
                hist = self.tech_indicators.add_all_indicators(hist)
                
                # Get current values
                current_price = hist['Close'][-1]
                current_volume = hist['Volume'][-1]
                avg_volume = hist['Volume'].mean()
                recent_high = hist['High'][-5:-1].max()
                
                # Check for breakout with volume and indicators
                if (current_price > recent_high and 
                    current_volume > 2 * avg_volume and 
                    current_price > Config.MIN_PRICE and
                    hist['SMA20'][-1] > hist['SMA50'][-1] and
                    hist['RSI'][-1] < 70):
                    
                    breakouts.append({
                        'symbol': symbol,
                        'price': current_price,
                        'breakout_level': recent_high,
                        'volume': current_volume,
                        'volume_ratio': round(current_volume / avg_volume, 2),
                        'sma20': hist['SMA20'][-1],
                        'sma50': hist['SMA50'][-1],
                        'rsi': round(hist['RSI'][-1], 2),
                        'time': hist.index[-1]
                    })
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        self.breakout_stocks = breakouts

    def get_sp500_symbols(self):
        """Get current S&P 500 symbols with caching"""
        cache_file = 'data/sp500_symbols.csv'
        try:
            # Try to read from cache first
            if os.path.exists(cache_file):
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if mtime.date() == self.today:
                    return pd.read_csv(cache_file)['Symbol'].tolist()
            
            # Fetch fresh data if cache is stale
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = table[0]
            df.to_csv(cache_file, index=False)
            return df['Symbol'].tolist()
        except:
            # Fallback if Wikipedia fails
            return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'QCOM']

    def run_scan(self):
        """Run complete scanning process"""
        print(f"\n{' Pre-Market ' if Config.is_pre_market() else ' Regular Market '} Scan - {datetime.now()}")
        
        # Step 1: Get potential symbols to scan
        symbols = self.get_sp500_symbols()
        
        # Step 2: Get pre-market movers
        self.get_premarket_movers()
        
        # Add gappers to our scan list
        if self.gappers:
            symbols += [item[0] for item in self.gappers]
        
        # Remove duplicates
        symbols = list(set(symbols))
        
        # Step 3: Run volume and breakout scans
        self.scan_unusual_volume(symbols)
        self.scan_breakouts(symbols)
        
        # Step 4: Display and alert results
        self.display_results()
        
        # Step 5: Send email alerts if any significant signals found
        if self.unusual_volume_stocks or self.breakout_stocks:
            self.send_alerts()

    def display_results(self):
        """Display scan results in console"""
        print("\n=== Pre-Market Gappers ===")
        for stock in self.gappers:
            print(f"{stock[0]}: {stock[1]} ({stock[2]}%) Volume: {stock[3]:,}")
        
        print("\n=== Unusual Volume Stocks ===")
        for stock in sorted(self.unusual_volume_stocks, key=lambda x: x['volume_ratio'], reverse=True):
            print(f"{stock['symbol']}: ${stock['price']} ({stock['price_change']}%) "
                  f"Volume: {stock['volume']:,} (Avg: {stock['avg_volume']:,.0f}, "
                  f"Ratio: {stock['volume_ratio']}x) @ {stock['time'].strftime('%H:%M')}")
        
        print("\n=== Breakout Candidates ===")
        for stock in sorted(self.breakout_stocks, key=lambda x: x['volume_ratio'], reverse=True):
            print(f"{stock['symbol']}: ${stock['price']} (Breakout: ${stock['breakout_level']}) "
                  f"Volume: {stock['volume']:,} (Ratio: {stock['volume_ratio']}x) "
                  f"SMA20/50: {stock['sma20']:.2f}/{stock['sma50']:.2f} RSI: {stock['rsi']}")

    def send_alerts(self):
        """Send email alerts for significant findings"""
        subject = f"Stock Scanner Alert - {len(self.unusual_volume_stocks)} Unusual Volume, {len(self.breakout_stocks)} Breakouts"
        
        body = "Stock Scanner Results:\n\n"
        body += "Unusual Volume Stocks:\n"
        for stock in self.unusual_volume_stocks:
            body += (f"{stock['symbol']}: ${stock['price']} ({stock['price_change']}%) "
                    f"Volume: {stock['volume']:,} (Ratio: {stock['volume_ratio']}x)\n")
        
        body += "\nBreakout Candidates:\n"
        for stock in self.breakout_stocks:
            body += (f"{stock['symbol']}: ${stock['price']} (Breakout: ${stock['breakout_level']}) "
                    f"Volume: {stock['volume']:,} (Ratio: {stock['volume_ratio']}x)\n")
        
        self.alert_system.send_alert(subject, body)

    def run_scheduled_scans(self):
        """Run scans on a schedule"""
        if Config.is_pre_market():
            # Run every 15 minutes during pre-market
            schedule.every(15).minutes.do(self.run_scan)
        else:
            # Run every 5 minutes during regular market hours
            schedule.every(5).minutes.do(self.run_scan)
        
        print("Scanner started. Press Ctrl+C to exit.")
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    scanner = StockScanner()
    
    # Run either once or continuously
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--continuous":
        scanner.run_scheduled_scans()
    else:
        scanner.run_scan()