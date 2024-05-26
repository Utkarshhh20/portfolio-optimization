#login
#create account
#create portfolio
#add portfolio
#track current portfolio

# https://www.financestrategists.com/wealth-management/investment-management/portfolio-optimization/
# https://www.financestrategists.com/uploads/Portfolio_Optimization_Methods.png
# Markowitz Mean-Variance Portfolio Optimization with Predictive Stock Selection Using Machine Learning
# there are two main issues of concern for practical application. The first is that the MV relies on the expected return and risk of asset inputs to produce optimal portfolios for each level of expected return and risk (Beheshti 2018). As a result, by selecting good assets to put into the optimization process, the MV model may achieve improved performance (Mitra Thakur et al. 2018). Another issue is that many high-risk assets often return a large number of small-scale weights in the optimal portfolio, which makes them difficult to implement, particularly for individual investors
# The method works by assuming investors are risk-averse. Specifically, it selects a set of assets that are least correlated (i.e., different from each other) and that generate the highest returns. This approach means that, given a set of portfolios with the same returns, you will select the portfolio with assets that have the least statistical relationship to one another.
# For example, instead of selecting a portfolio of tech company stocks, you should pick a portfolio with stocks across disparate industries. In practice, the mean variance optimization algorithm may select a portfolio containing assets in tech, retail, healthcare and real estate instead of a single industry like tech. Although this is a fundamental approach in modern portfolio theory, it has many limitations such as assuming that historical returns completely reflect future returns.
# Additional methods like hierarchical risk parity (HRP) and mean conditional value at risk (mCVAR) address some of the limitations of the mean variance optimization method. Specifically, HRP does not require inverting of a covariance matrix, which is a measure of how stock returns move in the same direction. The mean variance optimization method requires finding the inverse of the covariance matrix, however, which is not always computationally feasible.
# Further, the mCVAR method does not make the assumption that mean variance optimization makes, which happens when returns are normally distributed. Since mCVAR doesn’t assume normally distributed returns, it is not as sensitive to extreme values like mean variance optimization. This means that if a stock has an anomalous increase in price, mCVAR will be more robust than mean variance optimization and will be better suited for asset allocation. Conversely, mean variance optimization may naively suggest we disproportionately invest most of our resources in an asset that has an anomalous increase in price.
# The Black-Litterman model starts with an investor's views on the expected returns of different asset classes or securities and then uses these views to construct portfolios that maximize expected returns while minimizing risk.
# This model is particularly useful for investors who have strong views on the expected performance of specific asset classes or securities.
# Monte Carlo simulation is a method of portfolio optimization that uses random sampling to estimate the probability distribution of returns for different asset classes or securities.
# Monte Carlo simulation can be used to simulate the performance of different investment portfolios over time, which can help investors identify the optimal mix of assets for a given investment objective and risk tolerance.
# Importing libraries for portfolio optimization
from pypfopt import risk_models
from pypfopt import expected_returns
import pandas_datareader.data as web
import datetime
import pandas as pd 

#PORTFOLIO IS THE CLOSE OF ALL STOCKS WITHIN THE USERS PORTFOLIO

# Mean Variance Optimization
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

mu = mean_historical_return(portfolio)
S = CovarianceShrinkage(portfolio).ledoit_wolf()

# Here, we will use the max Sharpe statistic. The Sharpe ratio is the ratio between returns and risk. The lower the risk and the higher the returns, the higher the Sharpe ratio. The algorithm looks for the maximum Sharpe ratio, which translates to the portfolio with the highest return and lowest risk. Ultimately, the higher the Sharpe ratio, the better the performance of the portfolio. 
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights))

ef.portfolio_performance(verbose=True)

latest_prices = get_latest_prices(portfolio)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)

allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
# Invested decently heavily in multiple stocks but gave unrreal returns
# Of course, this return is inflated and is not likely to hold up in the future. 
# Mean variance optimization doesn’t perform very well since it makes many simplifying assumptions, such as returns being normally distributed and the need for an invertible covariance matrix. Fortunately, methods like HRP and mCVAR address these limitations. 

# Hierarchical Risk Parity (HRP)
# The HRP method works by finding subclusters of similar assets based on returns and constructing a hierarchy from these clusters to generate weights for each asset. 
from pypfopt import HRPOpt

returns = portfolio.pct_change().dropna()

hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()

hrp.portfolio_performance(verbose=True)
print(dict(hrp_weights))

da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=100000)

allocation, leftover = da_hrp.greedy_portfolio()
print("Discrete allocation (HRP):", allocation)
print("Funds remaining (HRP): ${:.2f}".format(leftover))
# Shown so much more diversification
#  Further, while the performance decreased, we can be more confident that this model will perform just as well when we refresh our data. This is because HRP is more robust to the anomalous increase in Moderna stock prices. 

# Mean Conditional Value at Risk (mCVAR)
# The mCVAR is another popular alternative to mean variance optimization. It works by measuring the worst-case scenarios for each asset in the portfolio, which is represented here by losing the most money. The worst-case loss for each asset is then used to calculate weights to be used for allocation for each asset. 
from pypfopt.efficient_frontier import EfficientCVaR

S = portfolio.cov()
ef_cvar = EfficientCVaR(mu, S)
cvar_weights = ef_cvar.min_cvar()

cleaned_weights = ef_cvar.clean_weights()
print(dict(cleaned_weights))

# We see that this algorithm suggests we invest heavily into JP Morgan Chase (JPM) and also buy a single share each of Moderna (MRNA) and Johnson & Johnson (JNJ). Also we see that the expected return is 15.5 percent. As with HRP, this result is much more reasonable than the inflated 225 percent returns given by mean variance optimization since it is not as sensitive to the anomalous behaviour of the Moderna stock price. 

# Black-Litterman Allocation
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting, objective_functions

mcaps = {}
for t in tickers:
    stock = yf.Ticker(t)
    mcaps[t] = stock.info["marketCap"]
mcaps

S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(market_prices)

plotting.plot_covariance(S, plot_correlation=True)
market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

market_prior.plot.barh(figsize=(10,5))

# Views
#In the BL method, views are specified via the matrix P (picking matrix) and the vector Q. Q contains the magnitude of each view, while P maps the views to the assets they belong to.
# If you are providing absolute views (i.e a return estimate for each asset), you don't have to worry about P and Q, you can just pass your views as a dictionary.
# You don't have to provide views on all the assets
# Black-Litterman also allows for relative views, e.g you think asset A will outperform asset B by 10%. If you'd like to incorporate these, you will have to build P and Q yourself. An explanation for this is given in the docs.
viewdict = {
    "AMZN": 0.10,
    "BAC": 0.30,
    "COST": 0.05,
    "DIS": 0.05,
    "DPZ": 0.20,
    "KO": -0.05,  # I think Coca-Cola will go down 5%
    "MCD": 0.15,
    "MSFT": 0.10,
    "NAT": 0.50,  # but low confidence, which will be reflected later
    "SBUX": 0.10
}

bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)
confidences = [
    0.6,
    0.4,
    0.2,
    0.5,
    0.7, # confident in dominos
    0.7, # confident KO will do poorly
    0.7, 
    0.5,
    0.1,
    0.4
]
bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict, omega="idzorek", view_confidences=confidences)
fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(bl.omega)

# We want to show all ticks...
ax.set_xticks(np.arange(len(bl.tickers)))
ax.set_yticks(np.arange(len(bl.tickers)))

ax.set_xticklabels(bl.tickers)
ax.set_yticklabels(bl.tickers)
plt.show()
np.diag(bl.omega)
# Note how NAT, which we gave the lowest confidence, also has the highest uncertainty.
# Instead of inputting confidences, we can calculate the uncertainty matrix directly by specifying 1 standard deviation confidence intervals, i.e bounds which we think will contain the true return 68% of the time. This may be easier than coming up with somewhat arbitrary percentage confidences

intervals = [
    (0, 0.25),
    (0.1, 0.4),
    (-0.1, 0.15),
    (-0.05, 0.1),
    (0.15, 0.25),
    (-0.1, 0),
    (0.1, 0.2),
    (0.08, 0.12),
    (0.1, 0.9),
    (0, 0.3)
]
variances = []
for lb, ub in intervals:
    sigma = (ub - lb)/2
    variances.append(sigma ** 2)

print(variances)
omega = np.diag(variances)

# Posterior estimates

# We are using the shortcut to automatically compute market-implied prior
bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta,
                        absolute_views=viewdict, omega=omega)
# Posterior estimate of returns
ret_bl = bl.bl_returns()
ret_bl
rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)], 
             index=["Prior", "Posterior", "Views"]).T
rets_df
rets_df.plot.bar(figsize=(12,8))
S_bl = bl.bl_cov()
plotting.plot_covariance(S_bl)
ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()
weights
pd.Series(weights).plot.pie(figsize=(10,10))

da = DiscreteAllocation(weights, prices.iloc[-1], total_portfolio_value=20000)
alloc, leftover = da.lp_portfolio()
print(f"Leftover: ${leftover:.2f}")
alloc

#ADDITIONAL STEPS
#EVALUATE OPTIMIZED PORTFOLIO
# Creating new portfolio with optimized weights
new_weights = LISTOFWEIGHTS
optimized_portfolio = STOCK*new_weights[0] + STOCK*new_weights[1]
optimized_portfolio # Visualizing daily returns

# Displaying new reports comparing the optimized portfolio to the first portfolio constructed
import quantstats as qs
qs.reports.full(optimized_portfolio, benchmark = portfolio)


# Advanced MVO - custom objectives
# If this new objective is convex, you can optimize a portfolio with the full benefit of PyPortfolioOpt's modular syntax, for example adding other constraints and objectives.
mu = expected_returns.capm_return(prices)
S = risk_models.semicovariance(prices)
import cvxpy as cp

# Note: functions are minimised. If you want to maximise an objective, stick a minus sign in it.
def logarithmic_barrier_objective(w, cov_matrix, k=0.1):
    log_sum = cp.sum(cp.log(w))
    var = cp.quad_form(w, cov_matrix)
    return var - k * log_sum
ef = EfficientFrontier(mu, S, weight_bounds=(0.01, 0.2))
ef.convex_objective(logarithmic_barrier_objective, cov_matrix=S, k=0.001)
weights = ef.clean_weights()
print(weights)
print(ef.portfolio_performance(verbose=True))