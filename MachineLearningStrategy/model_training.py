# model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap  # Requires `pip install shap`
import matplotlib.pyplot as plt

def train_model(X, y, model_type='RandomForest'):
    """
    Trains ML models to predict risk-adjusted returns.
    - Uses `scikit-learn` for model training (install via `pip install scikit-learn`).
    - Uses `shap` for feature importance analysis (install via `pip install shap`).
    """
    # Train-test split (time-series aware, no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Model selection
    if model_type == 'RandomForest':
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=5, 
            random_state=42
        )
    elif model_type == 'GradientBoosting':
        model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42
        )
    else:
        raise ValueError("Invalid model_type. Choose 'RandomForest' or 'GradientBoosting'.")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"[INFO] Model trained | MSE: {mse:.4f}")
    
    # SHAP analysis for interpretability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Save SHAP summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig("outputs/shap_plots/feature_importance.png", bbox_inches="tight")
    plt.close()
    
    return model