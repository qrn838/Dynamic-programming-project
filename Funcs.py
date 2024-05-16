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

# def marginal_utility(par,c,r):
#     """ marginal utility function """
#     if c>=r:
#        mu = 1/c * (1+par.eta)
#     else:
#        mu = 1/c * (1+par.eta*par.sigma)

#     return mu

def marginal_utility(par, c, r):
    """ Marginal utility function for array c """
    c = np.array(c)  # Ensure c is a NumPy array

    # Calculate marginal utility
    mu = np.where(c >= r, 1/c * (1 + par.eta), 1/c * (1 + par.eta * par.sigma))

    return mu

def inv_marg_utility_1(par,c):
    """ inverse marginal utility function """
    inv_mu = 1/c * (1+par.eta)
    return inv_mu

def inv_marg_utility_2(par,c):
    """ inverse marginal utility function """
    inv_mu = 1/c * (1+par.eta*par.sigma)
    return inv_mu
  

def cost(s):
	""" cost function """
	return 5*s**2


def inv_marg_cost(x):
    """ inverse marginal cost function """
    return 1/10*x