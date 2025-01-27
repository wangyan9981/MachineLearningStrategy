import pandas as pd
import numpy as np

def walk_forward_validation(data, model, train_window=252, test_window=63):
    sharpe_ratios = []
    max_drawdowns = []
    
    for i in range(0, len(data) - train_window - test_window, test_window):
        # Split data
        train_data = data.iloc[i:i+train_window]
        test_data = data.iloc[i+train_window:i+train_window+test_window]
        
        # Train model
        X_train, y_train = train_data.drop('target', axis=1), train_data['target']
        model.fit(X_train, y_train)
        
        # Predict returns
        X_test = test_data.drop('target', axis=1)
        predicted_returns = model.predict(X_test)
        
        # Convert to pandas Series for expanding() method
        predicted_returns = pd.Series(predicted_returns)
        cumulative_returns = (1 + predicted_returns).cumprod() - 1
        
        # Calculate drawdown
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (annualized)
        sharpe = predicted_returns.mean() / predicted_returns.std() * np.sqrt(252)
        
        sharpe_ratios.append(sharpe)
        max_drawdowns.append(max_drawdown)
    
    return np.mean(sharpe_ratios), np.mean(max_drawdowns)