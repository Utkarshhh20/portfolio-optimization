from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt.expected_returns import mean_historical_return
import pandas_datareader as pdr
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import expected_returns, EfficientSemivariance
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import streamlit as st
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import objective_functions
import plotly.express as px
import plotly.graph_objects as go
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import yfinance as yfin
from yahooquery import Screener
def plot_cum_returns(data, title):    
	daily_cum_returns = 1 + data.dropna().pct_change()
	daily_cum_returns = daily_cum_returns.cumprod()*100
	fig = px.line(daily_cum_returns, title=title)
	return fig
def plot_efficient_frontier_and_max_sharpe(mu, S): 
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S)
	fig, ax = plt.subplots(figsize=(6,4))
	ef_max_sharpe = copy.deepcopy(ef)
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig
start_date='2021-01-01'
end_date='2022-01-01'
tickers=['AAPL', 'MSFT', 'TSLA', 'GOOG', 'NVDA']
portfolio = pdr.get_data_yahoo(tickers, start = start_date, end = end_date)['Adj Close']
mu = mean_historical_return(portfolio)
S = portfolio.cov()
ef_cvar = EfficientCVaR(mu, S)
cvar_weights = ef_cvar.min_cvar()
cleaned_weights = ef_cvar.clean_weights()
weights_df = pd.DataFrame.from_dict(cleaned_weights, orient = 'index')
weights_df.columns = ['weights']  
        # Calculate returns of portfolio with optimized weights
portfolio['Optimized Portfolio'] = 0
for ticker, weight in cleaned_weights.items():
            portfolio['Optimized Portfolio'] += portfolio[ticker]*weight
print(ef_cvar.portfolio_performance)
print(dict(cleaned_weights))
latest_prices = get_latest_prices(portfolio)
da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=100000)
allocation, leftover = da_cvar.greedy_portfolio()
print("Discrete allocation (CVAR):", allocation)
print("Funds remaining (CVAR): ${:.2f}".format(leftover))
fig_cum_returns = plot_cum_returns(portfolio, '')
fig_cum_returns_optimized = plot_cum_returns(portfolio['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
st.plotly_chart(fig_cum_returns_optimized)
st.plotly_chart(fig_cum_returns)
st.write(cleaned_weights)

#####
#Semivariance
mu = expected_returns.mean_historical_return(portfolio)
historical_returns = expected_returns.returns_from_prices(portfolio)
es = EfficientSemivariance(mu, historical_returns)
es.efficient_return(0.20)

# We can use the same helper methods as before
weights = es.clean_weights()
print(weights)
weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
weights_df.columns = ['weights']  
        # Calculate returns of portfolio with optimized weights
portfolio['Optimized Portfolio'] = 0
for ticker, weight in weights.items():
            portfolio['Optimized Portfolio'] += portfolio[ticker]*weight
        
        # Plot Cumulative Returns of Optimized Portfolio
fig_cum_returns_optimized = plot_cum_returns(portfolio['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
latest_prices = get_latest_prices(portfolio)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)
allocation, leftover = da.greedy_portfolio()
print(es.portfolio_performance(verbose=True))
st.plotly_chart(fig_cum_returns_optimized)