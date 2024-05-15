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
<<<<<<< HEAD
    c = np.zeros(3)
    c[0] = par.cost1*s**(1+par.gamma)/(1+par.gamma)
    c[1] = par.cost2*s**(1+par.gamma)/(1+par.gamma)
    c[2] = par.cost3*s**(1+par.gamma)/(1+par.gamma)
=======
    
    if par.eta==0:
        c = np.zeros(3)
        c[0] = par.cost1*s**(1+par.gamma)/(1+par.gamma)
        c[1] = par.cost2*s**(1+par.gamma)/(1+par.gamma)
        c[2] = par.cost3*s**(1+par.gamma)/(1+par.gamma)
    else:
        c = np.zeros(2)
        c[0] = par.cost1*s**(1+par.gamma)/(1+par.gamma)
        c[1] = par.cost2*s**(1+par.gamma)/(1+par.gamma)
>>>>>>> 95851a81df8b26b6dd88eab38c21c4a13fc21371
    
    return c

def marg_cost(par,s):
    """ marginal cost from seaching function"""
<<<<<<< HEAD
    c_marg = np.zeros(3)
    c_marg[0] = par.cost1*s**(par.gamma)
    c_marg[1] = par.cost2*s**(par.gamma)
    c_marg[2] = par.cost3*s**(par.gamma)
=======

    if par.eta==0:
        c_marg = np.zeros(3)
        c_marg[0] = par.cost1*s**(par.gamma)
        c_marg[1] = par.cost2*s**(par.gamma)
        c_marg[2] = par.cost3*s**(par.gamma)
    else:
        c_marg = np.zeros(2)
        c_marg[0] = par.cost1*s**(par.gamma)
        c_marg[1] = par.cost2*s**(par.gamma)

>>>>>>> 95851a81df8b26b6dd88eab38c21c4a13fc21371
    return c_marg

def inv_marg_cost(par, s):
    """ inverse marginal cost from seaching function"""
<<<<<<< HEAD
    inv_c_marg = np.zeros(3)
    inv_c_marg[0] = (s/par.cost1)**(1/par.gamma)
    inv_c_marg[1] = (s/par.cost2)**(1/par.gamma)
    inv_c_marg[2] = (s/par.cost3)**(1/par.gamma)
=======
    
    if par.eta==0:
        inv_c_marg = np.zeros(3)
        inv_c_marg[0] = (s/par.cost1)**(1/par.gamma)
        inv_c_marg[1] = (s/par.cost2)**(1/par.gamma)
        inv_c_marg[2] = (s/par.cost3)**(1/par.gamma)
    else:
        inv_c_marg = np.zeros(2)
        inv_c_marg[0] = (s/par.cost1)**(1/par.gamma)
        inv_c_marg[1] = (s/par.cost2)**(1/par.gamma)

>>>>>>> 95851a81df8b26b6dd88eab38c21c4a13fc21371
    return inv_c_marg