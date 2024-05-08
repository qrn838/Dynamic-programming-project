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

def cost(par,s):
    """ cost from seaching function"""
    c = np.zeros(3)
    c[0] = par.cost[0]*s**(1+par.gamma)/(1+par.gamma)
    c[1] = par.cost[1]*s**(1+par.gamma)/(1+par.gamma)
    c[2] = par.cost[2]*s**(1+par.gamma)/(1+par.gamma)
    
    return c

def marg_cost(par,s):
    """ marginal cost from seaching function"""
    c_marg = np.zeros(3)
    c_marg[0] = par.cost[0]*s**(par.gamma)
    c_marg[1] = par.cost[1]*s**(par.gamma)
    c_marg[2] = par.cost[2]*s**(par.gamma)
    return c_marg

def inv_marg_cost(par, s):
    """ inverse marginal cost from seaching function"""
    inv_c_marg = np.zeros(3)
    inv_c_marg[0] = (s/par.cost[0])**(1/par.gamma)
    inv_c_marg[1] = (s/par.cost[1])**(1/par.gamma)
    inv_c_marg[2] = (s/par.cost[2])**(1/par.gamma)
    return inv_c_marg