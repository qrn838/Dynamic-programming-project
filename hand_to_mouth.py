import numpy as np
import copy
from scipy.optimize import minimize
from scipy.optimize import brentq
from Funcs import *

### Value function employment all periods ###

def value_function_employment_HTM(par, c, t):
    """ value function when employed """

    r = par.r_e_future[t,:]
    ref_diffs = np.zeros(par.N)
    for i in range(par.N):
         ref_diffs[i] = par.delta**(i+1) * (consumption_utility(c) - consumption_utility(r[i]))

    V = consumption_utility(c)/(1-par.delta)  + par.eta * np.sum(ref_diffs)

    return V



### Search effort and value function in steady state ###

# def unemployed_ss_HTM(par,i):

#     def objective_function(s, par, i):
#         V_e = value_function_employment_HTM(par, par.w, par.T - 1)
#         V_u = (consumption_utility(par.b4) - cost(par,s)[i] + par.delta * (s * V_e)) / (1-par.delta*(1-s))
#         return -V_u  # Minimize the negative of V_u to maximize V_u

#     # Perform optimization
#     s_initial_guess = 0.8
#     result = minimize(objective_function, s_initial_guess, args=(par,i,), bounds=[(0, 1)])

#     # Extract optimal s
#     optimal_s_ss = result.x[0]
#     V_u_ss = -result.fun

#     return optimal_s_ss, V_u_ss

def unemployed_ss_HTM(par,i):
    V_e = value_function_employment_HTM(par, par.w, par.T - 1)
    c = par.b4
    r = par.b4

    def bellman_difference(V_u):
        s = inv_marg_cost(par, par.delta * (V_e - V_u))[i]
        V_u_new = utility(par, c, r) - cost(par, s)[i] + par.delta * (s * V_e + (1 - s) * V_u)
        return V_u_new - V_u

    # Check values at the initial interval endpoints
    a, b = -1, 1
    fa = bellman_difference(a)
    fb = bellman_difference(b)

    # If the initial interval does not work, try finding a suitable interval by expanding
    if fa * fb > 0:
        interval_found = False

        # Generate search intervals dynamically with both positive and negative factors
        factors = [-50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50, 100, 200, 500]
        for factor_a in factors:
            for factor_b in factors:
                fa = bellman_difference(factor_a)
                fb = bellman_difference(-factor_b)
                if fa * fb < 0:
                    a, b = factor_a, factor_b
                    interval_found = True
                    break

        if not interval_found:
            raise ValueError("Could not find a valid interval where the function has different signs.")

    # Perform the root finding
    V_u = brentq(bellman_difference, a, b)
    s = inv_marg_cost(par, par.delta * (V_e - V_u))[i]

    return V_u, s



## Backward Induction to solve search effort in all periods of unemployment ###

def solve_search_effort_HTM(par):
    # a. allocate
    s = np.zeros((par.types,par.T))
    V_u = np.zeros((par.types,par.T))

    for i in range(par.types):

        # b. solve
        for t in range(par.T - 1, -1, -1):
            if t == par.T - 1:
                s[i,t], V_u[i,t] = unemployed_ss_HTM(par,i)
            
            else:
            
                V_e_next = value_function_employment_HTM(par, par.w, t+1)
                income = par.income_u[t]
                r = par.r_u[t]
                x = par.delta*(V_e_next-V_u[i,t+1])

                s[i,t] = inv_marg_cost(par,x)[i]
                V_u[i,t] = utility(par,income,r) - cost(par,s[i,t])[i] + par.delta * (s[i,t] * V_e_next+(1-s[i,t])*V_u[i,t+1])
    return s

def sim_search_effort_HTM(par):
    #Get policy functions
    s = solve_search_effort_HTM(par)

    type_shares = np.array([par.type_shares1, par.type_shares2, par.type_shares3])
    type_shares = type_shares[:par.types]

    """ Simulate search effort """
    s_sim = np.zeros((par.T))
    for t in range(par.T):
        if t == 0:
            s_sim[t] = type_shares @ s[:,t]  # search effort is weighted average of search efforts of types
        else:
            type_shares = type_shares*(1-s[:,t-1])  # update type shares as people get employed
            type_shares = type_shares/np.sum(type_shares) # normalize
            s_sim[t] = type_shares @ s[:,t]
    
    s_sim = s_sim[:par.T_sim]
    return s_sim



   