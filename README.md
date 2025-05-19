# Stock Scanner for Day Trading

A Python-based stock scanner that identifies:
- Pre-market gappers
- Unusual volume spikes
- Technical breakouts

## Features
- Scans during pre-market and regular trading hours
- Email alerts for significant findings
- Technical indicator analysis
- Backtesting capability

## Installation
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Create `.env` file from `.env.example` and add your API keys

## Usage
- Run once: `python scanner.py`
- Run continuously: `python scanner.py --continuous`

## Enhancements Implemented
1. Added technical indicators (SMA, RSI, MACD)
2. Incorporated Level 2 data through Polygon.io (configure in .env)
3. Added backtesting framework

## Configuration
Edit `config.py` or `.env` to:
- Set trading hours
- Adjust scan parameters
- Configure email alerts