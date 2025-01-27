import numpy as np
import pandas as pd

def create_features(data, window=30):
    """
    Generate technical features for each asset in multi-column data.
    """
    features = pd.DataFrame(index=data.index)
    returns = data.pct_change().dropna()
    
    # Iterate over each asset/ticker column
    for ticker in data.columns:
        # Momentum for the current ticker
        features[f'{ticker}_momentum'] = data[ticker].shift(1).rolling(window).mean()
        
        # Volatility for the current ticker
        features[f'{ticker}_volatility'] = returns[ticker].rolling(window).std()
        
        # RSI for the current ticker
        delta = data[ticker].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        features[f'{ticker}_RSI'] = 100 - (100 / (1 + gain / loss))
    
    return features.dropna()