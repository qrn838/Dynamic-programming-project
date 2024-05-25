import numpy as np
import copy
from scipy.optimize import minimize

from Funcs import *

### Value function employment all periods ###

def value_function_employment_HTM(par, c, t):
    """ value function when employed """

    r = par.r_e_future[t,:]         # reference points next N period from getting employed
    ref_diffs = np.zeros(par.N)     # Difference between future reference points and future consumption
    for i in range(par.N):
         ref_diffs[i] = par.delta**(i+1) * (consumption_utility(c) - consumption_utility(r[i]))

    V = consumption_utility(c)/(1-par.delta)  + par.eta * np.sum(ref_diffs) # value of getting employed at time t

    return V



### Search effort and value function in steady state ###

def unemployed_ss_HTM(par,i):
    """ Solve for search effort and value function in steady state when unemployed in HtM"""

    def objective_function(s, par, i):
        V_e = value_function_employment_HTM(par, par.w, par.T - 1)
        V_u = (consumption_utility(par.b4) - cost(par,s)[i] + par.delta * (s * V_e)) / (1-par.delta*(1-s))
        return -V_u  # Minimize the negative of V_u to maximize V_u

    # Perform optimization
    s_initial_guess = 0.8
    result = minimize(objective_function, s_initial_guess, args=(par,i,), bounds=[(0, 1)])

    # Extract optimal s
    optimal_s_ss = result.x[0]
    V_u_ss = -result.fun

    return optimal_s_ss, V_u_ss



## Backward Induction to solve search effort in all periods of unemployment ###

def solve_search_effort_HTM(par):
    """ Solve for search effort in all periods of unemployment in HtM"""
    s = np.zeros((par.types,par.T))
    V_u = np.zeros((par.types,par.T))

    for i in range(par.types):              # Loop over types

        for t in range(par.T - 1, -1, -1):  # Backward iteration
            if t == par.T - 1:              # Last period
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
    """ Simulate search effort in HtM"""

    s = solve_search_effort_HTM(par)

    type_shares = np.array([par.type_shares1, par.type_shares2, par.type_shares3])  # Array of types
    type_shares = type_shares[:par.types]  # Keep only the relevant types


    s_sim = np.zeros((par.T))
    for t in range(par.T):
        if t == 0:
            s_sim[t] = type_shares @ s[:,t]  # search effort is weighted average of search efforts of types
        else:
            type_shares = type_shares*(1-s[:,t-1])  # update type shares as people get employed
            type_shares = type_shares/np.sum(type_shares) # normalize
            s_sim[t] = type_shares @ s[:,t]     # search effort is weighted average of search efforts of types
    
    s_sim = s_sim[:par.T_sim]
    return s_sim



   