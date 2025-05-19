import os
from datetime import datetime, time
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
    FINNHUB_KEY = os.getenv('FINNHUB_KEY')
    POLYGON_KEY = os.getenv('POLYGON_KEY')
    
    # Email Settings
    EMAIL_SENDER = os.getenv('EMAIL_SENDER')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
    EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT')
    
    # Trading Hours
    PRE_MARKET_START = time(*map(int, os.getenv('PRE_MARKET_START', '04:00').split(':')))
    PRE_MARKET_END = time(*map(int, os.getenv('PRE_MARKET_END', '09:30').split(':')))
    REGULAR_MARKET_END = time(*map(int, os.getenv('REGULAR_MARKET_END', '16:00').split(':')))
    
    # Scan Parameters
    MIN_PRICE = 5.0
    MIN_VOLUME = 500000
    VOLUME_RATIO_THRESHOLD = 2.5
    PRICE_CHANGE_THRESHOLD = 1.5
    
    @staticmethod
    def is_pre_market():
        now = datetime.now().time()
        return Config.PRE_MARKET_START <= now < Config.PRE_MARKET_END
    
    @staticmethod
    def is_market_hours():
        now = datetime.now().time()
        return Config.PRE_MARKET_END <= now < Config.REGULAR_MARKET_END