#!/usr/bin/env python

"""
https://blog.quantopian.com/markowitz-portfolio-optimization-2/
"""

# STD
import json

# 3rd party
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import requests

import logging


def optimal_portfolio(returns):
    """
    @param returns numpy.maxtrix, one row per product, each value in the row represents one reading (price for a timestamp)

    @returns (weights, returns, risks)
        - weights weight of each asset in the portfolio
        - returns expected return for a particular risk (y-axis on the efficient frontier axis)
        - risks standard deviation for the portofolio (x-axis on the efficient frontier)
    """
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)


def random_portfolio(returns):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma



def test1():
    """
    Produces 500 random portfolios of (same) 4 assets each and shows the efficient frontier
    """
    np.random.seed(123)

    ## NUMBER OF ASSETS
    n_assets = 4

    ## NUMBER OF OBSERVATIONS
    n_obs = 1000

    return_vec = np.random.randn(n_assets, n_obs)

    n_portfolios = 500
    means, stds = np.column_stack([
        random_portfolio(return_vec)
        for _ in xrange(n_portfolios)
    ])

    weights, returns, risks = optimal_portfolio(return_vec)

    plt.plot(stds, means, 'o')
    plt.ylabel('mean')
    plt.xlabel('std')
    plt.plot(risks, returns, 'y-o')
    plt.plot(risks, returns)
    plt.show()



def get_real_stock_returns(tickerName):
    logging.info("Fetching {}".format(tickerName))
    token = "XA10GXCF7339XGUG"
    query = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}".format(tickerName, token)
    response = requests.get(query)
    data = json.loads(response.content)["Time Series (Daily)"]

    close_prices = [float(data[date]["4. close"]) for date in data]
    returns = [today - yesterday for yesterday, today in zip(close_prices, close_prices[1:])]
    return returns

def test2():
    """
    Using some real data now
    """

    tickers = ['IBM', 'TSLA', 'GOOG', 'AAPL', 'MSFT']
    stock_returns = np.matrix([get_real_stock_returns(ticker) for ticker in tickers])

    weights, returns, risks = optimal_portfolio(stock_returns)

    print weights

    plt.ylabel('mean')
    plt.xlabel('std')
    plt.plot(risks, returns, 'y-o')
    plt.plot(risks, returns)
    plt.show()


def init():
    # Turn off progress printing
    solvers.options['show_progress'] = False
    logging.basicConfig(level=logging.INFO, format='%(relativeCreated)6d %(message)s')

def main():
    init()

    test1()
    test2()


if __name__ == "__main__":
    main()
