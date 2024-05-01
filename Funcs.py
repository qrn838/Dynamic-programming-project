import numpy as np


def consumption_utility(x):
    """Instantaneous payoff function"""
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
	""" cost from seaching function"""
	return 5*s**2