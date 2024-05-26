from pypfopt import HRPOpt
import pandas_datareader as pdr
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
start_date='2021-01-01'
end_date='2022-01-01'
tickers=['AAPL', 'MSFT', 'TSLA', 'GOOG']
portfolio = pdr.get_data_yahoo(tickers, start = start_date, end = end_date)['Adj Close']
returns = portfolio.pct_change().dropna()
hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()
hrp.portfolio_performance(verbose=True)
print(dict(hrp_weights))
latest_prices = get_latest_prices(portfolio)
da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=100000)
allocation, leftover = da_hrp.greedy_portfolio()
print("Discrete allocation (HRP):", allocation)
print("Funds remaining (HRP): ${:.2f}".format(leftover))