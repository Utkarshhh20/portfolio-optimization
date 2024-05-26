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
st.set_page_config(page_title = "Bohmian's Stock Portfolio Optimizer", layout = "wide")
s = Screener()
tickers_strings = ''
count=0
sectordict={}
sectornames=['Individual Stocks', 'Technology', 'Utilities', 'Real Estate', 'Healthcare', 'Energy', 'Industrials', 'Materials', 'Communication Services', 'Financial Services', 'Consumer Defensive', 'Cryptocurrency']
sectors=['ms_technology', 'ms_utilities', 'ms_real_estate', 'ms_healthcare', 'ms_energy', 'ms_industrials', 'ms_basic_materials', 'ms_communication_services','ms_financial_services','ms_consumer_defensive', 'all_cryptocurrencies_us',]
portfolioinp=['Individual Stocks','ms_technology', 'ms_utilities', 'ms_real_estate', 'ms_healthcare', 'ms_energy', 'ms_industrials', 'ms_basic_materials', 'ms_communication_services','ms_financial_services','ms_consumer_defensive', 'all_cryptocurrencies_us']
for i in range(len(sectornames)):
    sectordict[sectornames[i]]=portfolioinp[i]

yfin.pdr_override()
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


st.header("Mean-variance Stock Portfolio Optimizer")

col1, col2, col3 = st.columns(3)

with col1:
	start_date = st.date_input("Start Date",datetime(2015, 1, 1))
	
with col2:
	end_date = st.date_input("End Date") # it defaults to current date
with col3:
    sectorinp=st.selectbox(label='Select a sector', options=sectornames, index=0)
data = s.get_screeners(sectors,  count=15)
for i in sectors:
    if i==sectordict[sectorinp]:
        df=pd.DataFrame(data[i]['quotes'])
        tickers=df['symbol']
        for j in tickers:
            if count!=0:
                tickers_strings = tickers_strings+','+j
            else:
                tickers_strings = tickers_strings+j
            count=count+1
    else:
        pass
if sectorinp=='Individual Stocks':
    tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas \
								WITHOUT spaces, e.g. "MA,FB,V,AMZN,JPM,BA"', '').upper()
else:
    tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas \
								WITHOUT spaces, e.g. "MA,FB,V,AMZN,JPM,BA"', value=tickers_strings).upper()
tickers = tickers_string.split(',')
try:
    # Get Stock Prices using pandas_datareader Library	
    stocks_df = pdr.get_data_yahoo(tickers, start = start_date, end = end_date)['Adj Close']
    sp500=pdr.get_data_yahoo('SPY', start = start_date, end = end_date)['Adj Close']
        # Plot Individual Stock Prices
    fig_price = px.line(stocks_df, title='')
        # Plot Individual Cumulative Returns
    fig_cum_returns = plot_cum_returns(stocks_df, '')
        # Calculatge and Plot Correlation Matrix between Stocks
    corr_df = stocks_df.corr().round(2)
    fig_corr = px.imshow(corr_df, text_auto=True)
        # Calculate expected returns and sample covariance matrix for portfolio optimization later
    mu = expected_returns.mean_historical_return(stocks_df)
    S = risk_models.sample_cov(stocks_df)
        
        # Plot efficient frontier curve
    fig = plot_efficient_frontier_and_max_sharpe(mu, S)
    fig_efficient_frontier = BytesIO()
    fig.savefig(fig_efficient_frontier, format="png")
        
        # Get optimized weights
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    ef.max_sharpe(risk_free_rate=0.02)
    weights = ef.clean_weights()
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
    weights_df.columns = ['weights']  
        # Calculate returns of portfolio with optimized weights
    stocks_df['Optimized Portfolio'] = 0
    for ticker, weight in weights.items():
            stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight
        
        # Plot Cumulative Returns of Optimized Portfolio
    fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
    latest_prices = get_latest_prices(stocks_df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)
    allocation, leftover = da.greedy_portfolio()
    print(allocation, leftover)
            # Display everything on Streamlit
    st.subheader("Your Portfolio Consists of: {} Stocks".format(tickers_string))
    col1,col2=st.columns([1.3,1])
    with col1:
        st.plotly_chart(fig_cum_returns_optimized, use_container_width=True)
    with col2:
        st.write('')	
        st.write('')	
        st.subheader('\tStock Prices')
        st.write(stocks_df)
    st.write('___________________________')
    col1,col2, stats=st.columns([0.5,1.3, 0.7])   
    with col1: 
        st.write('')
        st.write('')
        st.write('')
        st.subheader("Max Sharpe Portfolio Weights")
        st.dataframe(weights_df)
    with col2:
        st.write('')
        st.write('')
        stock_tickers=[]
        weightage=[]
        for i in weights:
            if weights[i]!=0:
                stock_tickers.append(i)
                weightage.append(weights[i])
        fig_pie = go.Figure(
            go.Pie(
            labels =stock_tickers,
            values = weightage,
            hoverinfo = "label+percent",
            textinfo = "value"
            ))
        holdings='''
        <style>
        .holding{
            float: center;
            font-weight: 600;
            font-size: 35px;
            font-family: arial;
        }
        </style>
        <body>
        <center><p1 class='holding'> Optimized Portfolio Holdings </p1></center>
        </body>
        '''
        st.markdown(holdings, unsafe_allow_html=True)
        st.plotly_chart(fig_pie)
    with stats:
        st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
        st.write('___________')
        st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
        st.write('___________')
        st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
        st.write('___________')
        st.subheader('''Discrete allocation: 
        {}'''.format(allocation))
        st.write('___________')
        st.subheader("Funds remaining: ${:.2f}".format(leftover))
    st.write('___________________________')
    col1, col2=st.columns(2)
    with col1:
        st.subheader("Optimized Max Sharpe Portfolio Performance")
        st.image(fig_efficient_frontier)
    with col2:
        st.subheader("Correlation between stocks")
        st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
    col1,col2=st.columns(2)
    with col1:
        st.subheader('Price of Individual Stocks')
        st.plotly_chart(fig_price)
    with col2:
        st.subheader('Cumulative Returns of Stocks Starting with $100')
        st.plotly_chart(fig_cum_returns)	
except:
	st.write('Enter correct stock tickers to be included in portfolio separated\
	by commas WITHOUT spaces, e.g. "MA,FB,V,AMZN,JPM,BA"and hit Enter.')
	
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 