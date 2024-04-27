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

def unemployed_ss(par):
    def objective_function(s, par):
        V_e = value_function_employment(par, par.w, par.T - 1)
        V_u = (consumption_utility(par.b3) - cost(s) + par.delta * (s * V_e)) / (1-par.delta*(1-s))
        return -V_u  # Minimize the negative of V_u to maximize V_u

    # Initial guess for s
    s_initial_guess = 0.8

    # Perform optimization
    result = minimize(objective_function, s_initial_guess, args=(par,), bounds=[(0, 1)])

    # Extract optimal s
    optimal_s_ss = result.x[0]
    V_u_ss = -result.fun

    return optimal_s_ss, V_u_ss


### Backward Induction to solve search effort in all periods of unemployment ###

def solve_search_effort(par):
    # a. allocate
    s = np.zeros(par.T)
    V_u = np.zeros(par.T)

    # b. solve
    for t in range(par.T - 1, -1, -1):
        if t == par.T - 1:
            def objective_function(s, par):
                V_e = value_function_employment(par, par.w, t)
                V_u = (consumption_utility(par.b3) - cost(s) + par.delta * (s * V_e+(1-s)*unemployed_ss(par)[1]))
                return -V_u
            
            s_initial_guess = unemployed_ss(par)[0]
            # Perform optimization
            result = minimize(objective_function, s_initial_guess, args=(par,), bounds=[(0, 1)])
            # Extract optimal s
            s[t] = result.x[0]
            V_u[t] = -result.fun
        
        else:
            def objective_function(s, par,t,V_u_next):
                V_e = value_function_employment(par, par.w, t)
                income = par.income_u[t]
                r = par.r_u[t]
                V_u = (utility(par,income,r) - cost(s) + par.delta * (s * V_e+(1-s)*V_u_next))
                return -V_u
            
            s_initial_guess = s[t+1]
            # Perform optimization
            result = minimize(objective_function, s_initial_guess, args=(par,t,V_u_next), bounds=[(0, 1)])
            # Extract optimal s
            s[t] = result.x[0]
            V_u[t] = -result.fun
        
        V_u_next = V_u[t]
        

    return s, V_u

   