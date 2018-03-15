#!/usr/bin/env python

"""
https://blog.quantopian.com/markowitz-portfolio-optimization-2/
"""

import numpy as np



def calculate_efficient_frontier(w, S, q, R):
    """
    For convenience, let's use this notation
    https://en.wikipedia.org/wiki/Modern_portfolio_theory

    - w is a vector of portfolio weights and sum(w) == 1
        (The weights can be negative, which means investors can short a security.);
    - S is the covariance matrix for the returns on the assets
    - q >= 0 is risk tolerance
        - q == 0 is risk averse
        - q == Inf is situatiaed maximally on the frontier
    - R is a vector of expected returns
    """
    pass


def test1():
    w = np.array([.3, .3, .4])
    S = np.matrix(
        [1, .3, .03],
        [.3, 1, .02],
        [.03, .02, 1],
    )
    q = 0
    R = np.array([5, 2, 10])

    result = calculate_efficient_frontier(w, S, q, R)
    print result


def test2():
    """
    http://www.calculatinginvestor.com/2011/06/07/efficient-frontier-1/
    """

    w = np.array([.25, .25, .25, .25])
    S = np.matrix(
        [185, 86.5, 80, 20],
        [86.5, 196, 76, 13.5],
        [80, 76, 411, -19],
        [20, 13.5, -19, 25],
    )
    q = 0
    R = np.array([14, 12, 15, 7])


def main():
    test1()
    test2()


if __name__ == "__main__":
    main()