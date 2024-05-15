import numpy as np
import copy
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import scipy.optimize as optimize

from Funcs import *

### Value function employment all periods ###

def value_function_employment(par, c, t):
    """ value function when employed """

    r = par.r_e_future[t,:]
    ref_diffs = np.zeros(par.N)
    for i in range(par.N):
         ref_diffs[i] = par.delta**(i+1) * (consumption_utility(c) - consumption_utility(r[i]))

    V = consumption_utility(c)/(1-par.delta)  + par.eta * np.sum(ref_diffs)

    return V



### Search effort and value function in steady state ###

def unemployed_ss(par,i):

    def objective_function(s, par, i):
        V_e = value_function_employment(par, par.w, par.T - 1)
        V_u = (consumption_utility(par.b4) - cost(par,s)[i] + par.delta * (s * V_e)) / (1-par.delta*(1-s))
        return -V_u  # Minimize the negative of V_u to maximize V_u

    # Perform optimization
    s_initial_guess = 0.8   # Arbitrary initial guess
    result = minimize(objective_function, s_initial_guess, args=(par,i), bounds=[(0, 1)])

    # Extract optimal s and value function for unemployed in steady state
    optimal_s_ss = result.x[0]
    V_u_ss = -result.fun

    return optimal_s_ss, V_u_ss

# Search effort unemployed SS

# def unemployed_ss(par, t, i):

#     V_e = value_function_employment(par, par.w, par.T - 1)
#     c = par.b4
#     r = par.r_u[t]

#     def bellman_difference(V_u):
#         s = inv_marg_cost(par.delta*(V_e-V_u))
#         V_u_new = (utility(par,c,r) - cost(s) + par.delta * (s * V_e + (1-s)*V_u)) 
        
#         return V_u_new - V_u

#     V_u = brentq(bellman_difference, -10, 0)
#     s = inv_marg_cost(par.delta*(V_e-V_u))

#     return V_u,s



## Backward Induction to solve search effort in all periods of unemployment ###

def solve_search_effort(par):
    # a. allocate
    s = np.zeros((par.types,par.T))
    V_u = np.zeros((par.types,par.T))

    for i in range(par.types):

        # b. solve
<<<<<<< HEAD
        for t in range(par.T - 1, -1, -1):
            if t == par.T - 1:
                s[i,t], V_u[i,t] = unemployed_ss(par, i)
=======
        for t in range(par.T - 1, -1, -1):    # Backward induction
            if t == par.T - 1:                # Last period
                s[i,t], V_u[i,t] = unemployed_ss(par,i)     # Use steady state values
>>>>>>> 95851a81df8b26b6dd88eab38c21c4a13fc21371
            
            else:
            
                V_e_next = value_function_employment(par, par.w, t+1)
                income = par.income_u[t]
                r = par.r_u[t]
                x = par.delta*(V_e_next-V_u[i,t+1])

                s[i,t] = inv_marg_cost(par,x)[i]
                V_u[i,t] = utility(par,income,r) - cost(par,s[i,t])[i] + par.delta * (s[i,t] * V_e_next+(1-s[i,t])*V_u[i,t+1])
    return s

def sim_search_effort(par):
    #Get policy functions
    s = solve_search_effort(par)

<<<<<<< HEAD
    type_shares = np.array([par.type_shares1, par.type_shares2, par.type_shares3])
    type_shares = type_shares[:par.types]

    """ Simulate search effort """
    s_sim = np.zeros((par.T))
    for t in range(par.T):
        if t == 0:
            s_sim[t] = type_shares @ s[:,t]  # search effort is weighted average of search efforts of types
        else:
            type_shares = type_shares*(1-s[:,t])  # update type shares as people get employed
            type_shares = type_shares/np.sum(type_shares) # normalize
            s_sim[t] = type_shares @ s[:,t]
=======
    if par.eta == 0:
        type_shares = np.array([par.type_shares1, par.type_shares2, par.type_shares3])      # 3 types for standard model with heterogeneous agents
    else:
        type_shares = np.array([par.type_shares1, par.type_shares2])                        # 2 types for model with reference depedence

    """ Simulate search effort """

    ### Not used (delete?) ###
    # s_sim = np.zeros((par.T))
    # for t in range(par.T):
    #     if t == 0:
    #         type_shares = type_shares[:par.types]
    #         s_sim[t] = type_shares @ s[:,t]  # search effort is weighted average of search efforts of types
    #     else:
    #         type_shares = type_shares*(1-s[:,t])  # update type shares as people get employed
    #         type_shares = type_shares/np.sum(type_shares) # normalize
    #         s_sim[t] = type_shares @ s[:,t]
    
    # s_sim = s_sim[:par.T_sim]


    ### Works ###
    s_sim = np.zeros(par.T_sim)
    for t in range(par.T_sim):
        if t == 0:
            # Initial search effort is the weighted average of search efforts of types
            s_sim[t] = type_shares @ s[:, t]  
        else:
            # Update type shares as people get employed
            type_shares *= (1 - s[:, t])
            # Normalize type shares
            type_shares /= np.sum(type_shares)
            # Calculate search effort for the next period
            s_sim[t] = type_shares @ s[:, t]


>>>>>>> 95851a81df8b26b6dd88eab38c21c4a13fc21371
    
    return s_sim



   