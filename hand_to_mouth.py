import numpy as np
import copy
from scipy.optimize import minimize

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
        V_u = (consumption_utility(par.b3) - cost(par,s)[i] + par.delta * (s * V_e)) / (1-par.delta*(1-s))
        return -V_u  # Minimize the negative of V_u to maximize V_u

    # Perform optimization
    s_initial_guess = 0.8
    result = minimize(objective_function, s_initial_guess, args=(par,i,), bounds=[(0, 1)])

    # Extract optimal s
    optimal_s_ss = result.x[0]
    V_u_ss = -result.fun

    return optimal_s_ss, V_u_ss



## Backward Induction to solve search effort in all periods of unemployment ###

def solve_search_effort(par):
    # a. allocate
    s = np.zeros((par.types,par.T))
    V_u = np.zeros((par.types,par.T))

    for i in range(par.types):

        # b. solve
        for t in range(par.T - 1, -1, -1):
            if t == par.T - 1:
                s[i,t], V_u[i,t] = unemployed_ss(par,i)
            
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

    if par.eta == 0:
        type_shares = np.array([par.type_shares1, par.type_shares2, par.type_shares3])
    else:
        type_shares = np.array([par.type_shares1, par.type_shares2])

    """ Simulate search effort """
    s_sim = np.zeros((par.T))
    for t in range(par.T):
        if t == 0:
            type_shares = type_shares[:par.types]
            s_sim[t] = type_shares @ s[:,t]  # search effort is weighted average of search efforts of types
        else:
            type_shares = type_shares*(1-s[:,t])  # update type shares as people get employed
            type_shares = type_shares/np.sum(type_shares) # normalize
            s_sim[t] = type_shares @ s[:,t]
    
    s_sim = s_sim[:par.T_sim]
    return s_sim



   