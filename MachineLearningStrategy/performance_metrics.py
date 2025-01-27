import pandas as pd

def calculate_performance(returns):
    """
    Computes Sharpe, Sortino, Calmar ratios, and win rate.
    """
    risk_free_rate = 0.02  # Assume 2% annualized
    
    # Sharpe Ratio
    sharpe = (returns.mean() - risk_free_rate/252) / returns.std() * np.sqrt(252)
    
    # Sortino Ratio (focuses on downside risk)
    downside_returns = returns[returns < 0]
    sortino = (returns.mean() - risk_free_rate/252) / downside_returns.std() * np.sqrt(252)
    
    # Calmar Ratio (returns vs max drawdown)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    calmar = returns.mean() / abs(drawdown.min())
    
    # Win Rate
    win_rate = len(returns[returns > 0]) / len(returns)
    
    return {
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Calmar': round(calmar, 2),
        'Win Rate': f"{win_rate:.1%}"
    }