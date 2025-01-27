import numpy as np

class RiskManager:
    def __init__(self, max_capital_risk=0.02, volatility_window=30):
        self.max_capital_risk = max_capital_risk
        self.volatility_window = volatility_window

    def calculate_position_size(self, price, volatility_series):
        """
        Calculate position size using rolling volatility (ATR).
        - `volatility_series`: Pandas Series of historical volatility.
        """
        recent_volatility = volatility_series.iloc[-self.volatility_window :]
        avg_volatility = np.mean(recent_volatility)
        dollar_risk = self.max_capital_risk * price
        position_size = dollar_risk / (avg_volatility * price)
        return round(position_size, 2)

    def trailing_stop_loss(self, entry_price, current_price, atr, multiplier=2):
        """
        Trailing stop based on 2x ATR.
        """
        if current_price > entry_price:
            new_stop = current_price - multiplier * atr
            return max(new_stop, entry_price - multiplier * atr)
        return entry_price - multiplier * atr

  