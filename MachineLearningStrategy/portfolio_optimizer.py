import cvxpy as cp  

def optimize_portfolio(expected_returns, covariance_matrix, target_return=0.15):
    n_assets = len(expected_returns)
    weights = cp.Variable(n_assets)
    
    portfolio_variance = cp.quad_form(weights, covariance_matrix)
    objective = cp.Minimize(portfolio_variance)
    
    constraints = [
        expected_returns @ weights >= target_return,
        cp.sum(weights) == 1,
        weights >= 0
    ]
    
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.SCS)  # Use SCS instead of ECOS
        if weights.value is None:
            raise ValueError("Optimization failed: Infeasible or unbounded problem.")
        return weights.value
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        return None