import pandas as pd
from utils.technical_indicators import TechnicalIndicators

class Backtester:
    def __init__(self):
        self.tech = TechnicalIndicators()
    
    def backtest_strategy(self, symbol, strategy='breakout', period='1y'):
        """Backtest a trading strategy on historical data"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            hist = self.tech.add_all_indicators(hist)
            
            if strategy == 'breakout':
                return self._test_breakout_strategy(hist)
            elif strategy == 'volume_spike':
                return self._test_volume_spike_strategy(hist)
            else:
                return {"error": "Unknown strategy"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _test_breakout_strategy(self, hist):
        """Test breakout strategy"""
        signals = []
        position = 0
        pnl = 0
        trades = []
        
        for i in range(1, len(hist)):
            # Breakout signal
            if (hist['Close'][i] > hist['High'][i-1] and 
                hist['Volume'][i] > 1.5 * hist['Volume'][i-1] and 
                hist['SMA20'][i] > hist['SMA50'][i]):
                
                if position <= 0:
                    entry_price = hist['Close'][i]
                    signals.append(('buy', hist.index[i], entry_price))
                    if position < 0:
                        # Close short
                        pnl += (trades[-1][1] - entry_price)
                        trades[-1] = (trades[-1][0], hist.index[i], trades[-1][2], entry_price, pnl)
                    # Open long
                    position = 1
                    trades.append(('long', hist.index[i], entry_price, None, None))
            
            # Exit signal
            elif position > 0 and (hist['Close'][i] < hist['SMA20'][i] or 
                                  hist['RSI'][i] > 70):
                exit_price = hist['Close'][i]
                signals.append(('sell', hist.index[i], exit_price))
                pnl += (exit_price - trades[-1][2])
                trades[-1] = (trades[-1][0], trades[-1][1], trades[-1][2], hist.index[i], pnl)
                position = 0
        
        return {
            "signals": signals,
            "trades": trades,
            "total_pnl": pnl,
            "win_rate": self._calculate_win_rate(trades)
        }
    
    def _test_volume_spike_strategy(self, hist):
        """Test volume spike strategy"""
        # Similar implementation for volume strategy
        pass
    
    def _calculate_win_rate(self, trades):
        if not trades:
            return 0
        wins = sum(1 for t in trades if t[4] and t[4] > 0)
        return wins / len(trades)