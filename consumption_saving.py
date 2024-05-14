import numpy as np
import copy
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import scipy.optimize as optimize
import numba

from Funcs import *


### Value of getting employed ###

def objective_function(a_next, par, i_a, i_t, i_n, V_e_next):
    r = par.r_e_m[i_t, i_t + i_n]
    c = par.a_grid[i_a] + par.w - a_next / (par.R)
    V_e_next_interp = interp1d(par.a_grid, V_e_next)
    V_e = utility(par, c, r) + par.delta * V_e_next_interp(a_next)
    return -V_e

def value_function_employment(par, sol):
    """Value function when employed"""
    V_e = np.zeros((par.T, par.N+par.M, par.Na))
    V_e_next = np.zeros((par.T, par.N+par.M, par.Na))   # M to ensure that we have converged to stationary state. Check this is the case
    a_next = np.zeros((par.T, par.N+par.M, par.Na))
    m_egm = np.zeros((par.T, par.N+par.M, par.Na))
    c_egm = np.zeros((par.T, par.N+par.M, par.Na))

    for i_t in range(par.T): 
        for i_n in range(par.N+par.M-1, -1, -1):
            for i_a in range(par.Na):
                

                if i_n == par.N+par.M - 1:  # Stationary state
                    r = par.r_e_m[i_t, i_t + i_n]
                    c = par.a_grid[i_a] + par.w - par.a_grid[i_a] / (par.R)
                    V_e[i_t, i_n, i_a] = utility(par, c, r) / (1 - par.delta)  ## stationary state
                    a_next[i_t, i_n, i_a] = par.a_grid[i_a]

                    m_egm[i_t, i_n, :] = par.a_grid[:] + par.w
                    c_egm[i_t, i_n, i_a] = par.a_grid[i_a] + par.w - par.a_grid[i_a] / (par.R)


                else: # Cosumption saving problem

                    if par.euler == False:  # With optimizer
                        a_next_guess = a_next[i_t, i_n+1, i_a]
                        lower_bound = par.L
                        upper_bound = (par.a_grid[i_a] + par.w)*par.R - 10e-6  # consumption must be positive
                        upper_bound = min(upper_bound, par.A_0)
                        result = minimize(objective_function, a_next_guess, args=(par, i_a, i_t, i_n, V_e_next[i_t, i_n+1, :]), method='nelder-mead', bounds=[(lower_bound, upper_bound)])  #SLSQP
                        if result.success:
                            a_next[i_t, i_n, i_a] = result.x[0]
                            V_e[i_t, i_n, i_a] = -result.fun
                            
                        else:
                            print("Error at t={}, i_n={}, i_a={}".format(i_t, i_n, i_a))
                            print("Error message:", result.message)
                    
                    elif par.euler == True:  # EGM
                        # a_grid is now seen as next period assets
                        m = par.a_grid[:] + par.w  #exo grid. How much cash on hand you start the period with
                        c_next_int = interp1d(m,c_egm[i_t,i_n+1,:])
                        c_next = c_next_int(m_egm[i_t,i_n+1,i_a])

                        marg_util_next = par.delta * par.R * marginal_utility(par, c_next, par.r_e_m[i_t, i_t + i_n+1])
                        c1 = inv_marg_utility_1(par, marg_util_next)
                        c2 = inv_marg_utility_2(par, marg_util_next)

                        m1 = par.a_grid[i_a]/par.R + c1 # chash on hand needen this period to obtain c and next period assets
                        m2 = par.a_grid[i_a]/par.R + c2

                        #next period savings as function of current cash-on-hand
                        a1 = interp1d(m, par.a_grid, fill_value="extrapolate")(m1)
                        a2 = interp1d(m, par.a_grid, fill_value="extrapolate")(m2)

                        # constraints
                        if m1 > par.w:
                            c1 = par.w - par.a_grid[i_a] / (par.R)
                            m1 = par.w
                            a1 = interp1d(m, par.a_grid)(m1)
                        if m2 > par.w:
                            c2 = par.w - par.a_grid[i_a] / (par.R)
                            m2 = par.w
                            a2 = interp1d(m, par.a_grid)(m2)

                  
                        if c1 >= par.r_e_m[i_t, i_t + i_n] and c2 >= par.r_e_m[i_t, i_t + i_n]:
                            c_egm[i_t, i_n, i_a] = c1
                            m_egm[i_t, i_n, i_a] = m1
                            a_next[i_t, i_n, i_a] = a1

                        elif c1 <= par.r_e_m[i_t, i_t + i_n] and c2 <= par.r_e_m[i_t, i_t + i_n]:
                            c_egm[i_t, i_n, i_a] = c2
                            m_egm[i_t, i_n, i_a] = m2
                            a_next[i_t, i_n, i_a] = a2
                        
                        else:
                            V_e_1 = utility(par, c1, par.r_e_m[i_t, i_t + i_n]) + par.delta * interp1d(par.a_grid,V_e_next[i_t, i_n+1, :], fill_value="extrapolate")(a1)

                            V_e_2 = utility(par, c2, par.r_e_m[i_t, i_t + i_n]) + par.delta * interp1d(par.a_grid,V_e_next[i_t, i_n+1, :], fill_value="extrapolate")(a2)

                            if V_e_1 > V_e_2:
                                c_egm[i_t, i_n, i_a] = c1
                                m_egm[i_t, i_n, i_a] = m1
                                a_next[i_t, i_n, i_a] = a1
                            else:
                                c_egm[i_t, i_n, i_a] = c2   
                                m_egm[i_t, i_n, i_a] = m2
                                a_next[i_t, i_n, i_a] = a2                            

                        V_e[i_t, i_n, i_a] = utility(par, c_egm[i_t, i_n, i_a], par.r_e_m[i_t, i_t + i_n]) + par.delta * interp1d(par.a_grid,V_e_next[i_t, i_n+1, :])(a_next[i_t, i_n, i_a])

                V_e_next[i_t, i_n, i_a] = V_e[i_t, i_n, i_a]
                
        
    par.V_e_t_a = V_e[:, 0, :]
    par.V_e = V_e
    sol.a_next_e = a_next



# Search effort unemployed SS

def unemployment_ss(par, t, i_a):
    V_e = par.V_e_t_a[t, i_a]
    c = par.a_grid[i_a] + par.income_u[t] - par.a_grid[i_a] / (par.R)
    r = par.r_u[t]

    def bellman_difference(V_u):
        s = inv_marg_cost(par.delta*(V_e-V_u))
        V_u_new = (utility(par,c,r) - cost(s) + par.delta * (s * V_e + (1-s)*V_u)) 
        
        return V_u_new - V_u

    V_u = brentq(bellman_difference, -10, 0)
    s = inv_marg_cost(par.delta*(V_e-V_u))

    return V_u,s




### Backward Induction to solve search effort in all periods of unemployment ###

def solve_search_and_consumption(par, sol):
    # a. allocate
    tuple = (par.T, par.Na)
    s = np.zeros(tuple)
    V_u = np.zeros(tuple)
    V_u_next = np.zeros(tuple)

    c = np.zeros(tuple)
    a_next = np.zeros(tuple)


    # b. solve
    for t in range(par.T - 1, -1, -1):
        for i_a in range(par.Na):
            if t == par.T - 1:   # Stationary state
                V_u[t,i_a] = unemployment_ss(par,t, i_a)[0]
                s[t,i_a] = unemployment_ss(par, t, i_a)[1]
                a_next[t,i_a] = par.a_grid[i_a]
            
            else: # Previous periods. Chech that debt converges to par.L before stationary state when solving forward
                def objective_function_ti(a_next, par,t,V_u_next):
                    
                    income = par.income_u[t]
                    r = par.r_u[t]
                    c = par.a_grid[i_a] + income - a_next / (par.R)
                    V_e_next_interp = interp1d(par.a_grid, par.V_e_t_a[t+1, :])
                    V_e_next = V_e_next_interp(a_next)
                    V_u_next_interp = interp1d(par.a_grid, V_u_next)
                    V_u_next = V_u_next_interp(a_next)
                    s = inv_marg_cost(par.delta*(V_e_next-V_u_next))
                    V_u = utility(par,c,r) - cost(s) + par.delta * (s * V_e_next+(1-s)*V_u_next)
                    return -V_u
                
                a_next_guess = par.a_grid[i_a]

                lower_bound = par.L
                upper_bound = (par.a_grid[i_a] + par.income_u[t])*par.R - 10e-6
                upper_bound = min(upper_bound, par.A_0)
                result = minimize(objective_function_ti, a_next_guess, args=(par, t, V_u_next[t+1,:]), method='nelder-mead', bounds=[(lower_bound, upper_bound)])

                             
                # Extract optimal a_next
                if result.success:
                                      
                    a_next[t, i_a] = result.x[0]
                    V_u[t,i_a] = -result.fun
                    
                    V_e_next_interp = interp1d(par.a_grid, par.V_e_t_a[t+1, :])
                    V_e_next = V_e_next_interp(a_next[t,i_a])
                    V_u_next_interp = interp1d(par.a_grid, V_u_next[t+1,:])
                    V_u_next_int = V_u_next_interp(a_next[t, i_a])
                    s[t,i_a] = inv_marg_cost(par.delta*(V_e_next-V_u_next_int))
                    c[t,i_a] = par.a_grid[i_a] + par.income_u[t] - a_next[t,i_a] / (par.R)
              
                else:
                    print("Error at t={}, i_a={}".format(t, i_a))
                    print("Error message:", result.message)

            V_u_next[t,i_a] = V_u[t,i_a]
        
    sol.s = s
    sol.a_next = a_next



### Solve forward from initial assets to get true search and consumption path ###

def solve_forward(par, sol, sim):
    s = np.zeros(par.T)
    a_next = np.zeros(par.T)
    # b. solve
    for t in range(par.T):
        if t == 0: # First period
            a = par.A_0
            s[t] = sol.s[t,-1]
            a_next[t] = sol.a_next[t,-1]
        else:
            a = a_next[t-1]
            s_interp = interp1d(par.a_grid, sol.s[t, :])
            a_next_interp = interp1d(par.a_grid, sol.a_next[t, :])
            s[t] = s_interp(a)
            a_next[t] = a_next_interp(a)
    
    sim.s = s
    sim.a_next = a_next


def solve_forward_employment(par, sol, sim):

    a_next = np.zeros((par.T, par.N+par.M+1))
    # b. solve
    for t in range(par.T):
        for n in range(par.N+par.M):
            if t == 0: # First period
                if n == 0:
                    a_next[t, n] = par.A_0

                if n == 1:
                    a = par.A_0
                    a_next[t, n] = sol.a_next_e[t, n, -1]
                else:
                    a = a_next[t, n-1]
                    a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :])
                    a_next[t, n] = a_next_interp(a)
            else:
                if n == 0:
                    a = sim.a_next[t-1]
                    a_next[t, n] = sim.a_next[t-1]

                elif n == 1:
                    a = sim.a_next[t-1]
                    a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :])
                    a_next[t, n] = a_next_interp(a)
                else:
                    a = a_next[t, n-1]
                    a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :])
                    a_next[t, n] = a_next_interp(a)
    sim.a_e = a_next



