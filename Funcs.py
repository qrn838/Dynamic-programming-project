import numpy as np


def consumption_utility(x):
	v = np.log(x)
	return v


def utility(par,c,r):
    """ utility function """
	# a. utility
    if c>=r:
       u = consumption_utility(c) + par.eta*(consumption_utility(c)-consumption_utility(r))
    else:
       u = consumption_utility(c) + par.eta*par.sigma*(consumption_utility(c)-consumption_utility(r))

    return u

def cost(s):
	""" cost function """
	return 5*s**2


def inv_marg_cost(x):
    """ inverse marginal cost function """
    return 1/10*x