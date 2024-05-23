import numpy as np
import copy
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import scipy.optimize as optimize
from scipy.optimize import bisect
import numba

from Funcs import *
def value_function_employment(par, sol):
    if par.euler == False:
        value_function_employment_VFI(par, sol)
    else:
        value_function_employment_EGM(par, sol)


########## VFI #########
def objective_function(a_next, par, i_a, i_t, i_n, V_e_next):
    r = par.r_e_m[i_t, i_t + i_n]
    c = par.a_grid[i_a] + par.w - a_next / (par.R)
    penalty = 0
    if c < 1e-6:
        penalty = 1e8*(c)**2

    V_e_next_interp = interp1d(par.a_grid, V_e_next)
    V_e = utility(par, c, r) + par.delta * V_e_next_interp(a_next) - penalty
    return -V_e

def value_function_employment_VFI(par, sol):
    """Value function when employed"""
    V_e = np.zeros((par.T, par.N + par.M, par.Na))
    a_next = np.zeros((par.T, par.N + par.M, par.Na))
    c = np.zeros((par.T, par.N + par.M, par.Na))

    for i_t in range(par.T): 
        for i_n in reversed(range(par.N + par.M)):
            for i_a in range(par.Na):
                if i_n == par.N + par.M - 1:  # Stationary state
                    r = par.r_e_m[i_t, i_t + i_n]
                    # c[i_t, i_n, i_a] = par.a_grid[i_a] + par.w - par.a_grid[i_a] / par.R
                    # a_next[i_t, i_n, i_a] = par.a_grid[i_a]

                    c[i_t, i_n, i_a] = par.a_grid[i_a] + par.w                   
                    a_next[i_t, i_n, i_a] = (par.a_grid[i_a] + par.w -c[i_t, i_n, i_a])*par.R

                    V_e[i_t, i_n, i_a] = utility(par, c[i_t, i_n, i_a], r) / (1 - par.delta)  ## stationary state
                    
                    

                else:  # Consumption saving problem
                    lower_bound = par.L
                    upper_bound = (par.a_grid[i_a] + par.w) * par.R - 1e-6  # consumption must be positive
                    upper_bound = min(upper_bound, par.A_0)

                    # Run the optimizer using minimize_scalar with method='golden'
                    result = minimize_scalar(objective_function, bounds=(lower_bound, upper_bound), args=(par, i_a, i_t, i_n, V_e[i_t, i_n + 1, :]), method='bounded')

                    if result.success:
                        a_next[i_t, i_n, i_a] = result.x
                        V_e[i_t, i_n, i_a] = -result.fun
                        c[i_t, i_n, i_a] = par.a_grid[i_a] + par.w - a_next[i_t, i_n, i_a] / par.R
                    else:
                        print(f"Error at t={i_t}, i_n={i_n}, i_a={i_a}")
                        print("Error message:", result.message)
                        print("Current function value:", result.fun)
                        print("Optimization result details:", result)
                
            

    par.V_e_t_a = V_e[:, 0, :]
    par.V_e = V_e
    sol.a_next_e = a_next
    sol.c_e = c



############## EGM #####################

def value_function_employment_EGM(par, sol):
    """Value function when employed"""
    V_e = np.zeros((par.T, par.N + par.M, par.Na))
    a_next = np.zeros((par.T, par.N + par.M, par.Na))
    a_egm = np.zeros((par.T, par.N + par.M, par.Na))
    c_egm = np.zeros((par.T, par.N + par.M, par.Na))

    for i_t in range(par.T):
        for i_a in range(par.Na):
            r = par.r_e_m[i_t, -1]
            c = par.a_grid[i_a] + par.w - par.a_grid[i_a] / par.R
            V_e[i_t, -1, i_a] = utility(par, c, r) / (1 - par.delta)  ## stationary state
            a_next[i_t, -1, i_a] = par.a_grid[i_a]

            a_egm[i_t, -1, i_a] = par.a_grid[i_a]/par.R + c - par.w  # Cash on hand needed this period to obtain c and next period assets
            c_egm[i_t, -1, i_a] = par.a_grid[i_a] + par.w - par.a_grid[i_a] / par.R

        
    for i_t in range(par.T):
        for i_n in range(par.N + par.M - 2, -1, -1):
            
            c_next_int = interp1d(a_egm[i_t, i_n+1,:],c_egm[i_t, i_n+1,:], fill_value='extrapolate')
            c_next = c_next_int(par.a_grid)

            marg_util_next = marginal_utility(par, c_next, par.r_e_m[i_t, i_t + i_n + 1])
            c1 = inv_marg_utility_1(par, par.delta * par.R *marg_util_next)
            c2 = inv_marg_utility_2(par, par.delta * par.R *marg_util_next)
         

            a1 = par.a_grid[:] / par.R + c1 - par.w  # Cash on hand needed this period to obtain c and next period assets
            a2 = par.a_grid[:] / par.R + c2 - par.w
            # print(a1-par.a_grid)
            # print(a2-par.a_grid)

            # #next period assets as function of current period assets
            # a1_next = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a1)
            # a2_next = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a2)
        
            # for i in range(par.Na):
            #     if a1[i] > par.A_0:
            #         a1[i] = par.A_0
            #         c1[i] = a1[i] + par.w - par.a_grid[i] / par.R
            #         # a1_next[i] = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a1[i])

            #     if a2[i] > par.A_0:
            #         a2[i] = par.A_0
            #         c2[i] = a2[i] + par.w - par.a_grid[i] / par.R
                    # a2_next[i] = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a2[i])
                    
                # if a1[i] < par.L:
                #     a1[i] = par.L
                #     c1[i] = a1[i] + par.w - par.a_grid[i] / par.R
                #     # a1_next[i] = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a1[i])
                # if a2[i] < par.L:
                #     a2[i] = par.L
                #     c2[i] = a2[i] + par.w - par.a_grid[i] / par.R
                #     # a2_next[i] = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a2[i])


            # a1_next = interp1d(a1, par.a_grid, fill_value="extrapolate")(par.a_grid)
            # a2_next = interp1d(a2, par.a_grid, fill_value="extrapolate")(par.a_grid)
            # print(a1_next)

            for i in range(par.Na):
                
                # if c1[i] >= par.r_e_m[i_t, i_t + i_n] and c2[i] >= par.r_e_m[i_t, i_t + i_n]:
                if 2 > 1:
                    c_egm[i_t, i_n, i] = c1[i]
                    a_egm[i_t, i_n, i] = a1[i]
                    # a_egm[i_t, i_n, i] = a1[i]
                    
                    
                elif  c1[i] < par.r_e_m[i_t, i_t + i_n] and c2[i] < par.r_e_m[i_t, i_t + i_n]:
                # elif 2 > 1:
                    c_egm[i_t, i_n, i] = c2[i]
                    a_egm[i_t, i_n, i] = a2[i]
                    # a_egm[i_t, i_n, i] = a2[i]
                    
            
                else:
                    # print('Error in EGM at time period', i_t, 'and state', i_n, 'and asset', i)
                    V_e_1 = utility(par, c1[i], par.r_e_m[i_t, i_t + i_n]) + par.delta * interp1d(a_egm[i_t, i_n + 1, :], V_e[i_t, i_n + 1, :], fill_value="extrapolate")(par.a_grid[i])
                    V_e_2 = utility(par, c2[i], par.r_e_m[i_t, i_t + i_n]) + par.delta * interp1d(a_egm[i_t, i_n + 1, :], V_e[i_t, i_n + 1, :], fill_value="extrapolate")(par.a_grid[i])

                    if V_e_1 > V_e_2:
                        c_egm[i_t, i_n, i] = c1[i]
                        a_egm[i_t, i_n, i] = a1[i]
                        # a_next[i_t, i_n, i] = a1_next[i]
                    
                    else:
                        c_egm[i_t, i_n, i] = c2[i] 
                        a_egm[i_t, i_n, i] = a2[i]
                        # a_next[i_t, i_n, i]= a2_next[i]

                V_e[i_t, i_n, i] = utility(par, c_egm[i_t, i_n, i], par.r_e_m[i_t, i_t + i_n]) + par.delta * interp1d(a_egm[i_t, i_n + 1, :], V_e[i_t, i_n + 1, :], fill_value="extrapolate")(par.a_grid[i])

             ## Interp to get as function of current assets            
            for i_a in range(par.Na):
                if i_a == 0:
                    print(a_egm[i_t, i_n, :])
                    print(par.a_grid)
                # print(c_egm[i_t, i_n, i_a])
                a_next[i_t, i_n, i_a] = interp1d(a_egm[i_t, i_n, :], par.a_grid, fill_value="extrapolate")(par.a_grid[i_a])
                V_e[i_t, i_n, i_a] = interp1d(a_egm[i_t, i_n, :], V_e[i_t, i_n, :], fill_value="extrapolate")(par.a_grid[i_a])
                c_egm[i_t, i_n, i_a] = interp1d(a_egm[i_t, i_n, :], c_egm[i_t, i_n, :], fill_value="extrapolate")(par.a_grid[i_a])
                # print(c_egm[i_t, i_n, i_a])
    par.V_e_t_a = V_e[:, 0, :]
    par.V_e = V_e
    sol.a_next_e = a_next
    sol.c_e = c_egm


# Search effort unemployed SS

# def unemployment_ss(par, t, i_a):
#     V_e = par.V_e_t_a[t, i_a]
#     c = par.a_grid[i_a] + par.income_u[t] - par.a_grid[i_a] / (par.R)
#     r = par.r_u[t]

#     def bellman_difference(V_u):
#         s = inv_marg_cost(par.delta*(V_e-V_u))
#         V_u_new = (utility(par,c,r) - cost(s) + par.delta * (s * V_e + (1-s)*V_u)) 
        
#         return V_u_new - V_u
    
#     a = -20
#     b = 10
#     V_u = bisect(bellman_difference, a, b)
#     #V_u = brentq(bellman_difference, -5, 5)
#     s = inv_marg_cost(par.delta*(V_e-V_u))

#     return V_u,s



def unemployment_ss(par,t, i_a):
    """ Solve for search effort and value function in steady state when unemployed"""
    V_e = par.V_e_t_a[t, i_a]
    c = par.a_grid[i_a] + par.income_u[t] - par.a_grid[i_a] / (par.R)
    r = par.r_u[t]

    def objective_function(s, par, V_e, c, r):
        V_u = (utility(par, c, r) - cost(s) + par.delta * (s * V_e)) / (1-par.delta*(1-s))
        return -V_u  # Minimize the negative of V_u to maximize V_u

    # Perform optimization
    s_initial_guess = 0.8
    result = minimize(objective_function, s_initial_guess, args=(par,V_e, c, r), bounds=[(0, 1)])

    # Extract optimal s
    optimal_s_ss = result.x[0]
    V_u_ss = -result.fun

    return optimal_s_ss, V_u_ss




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
                V_u[t,i_a] = unemployment_ss(par,t, i_a)[1]
                s[t,i_a] = unemployment_ss(par, t, i_a)[0]
                a_next[t,i_a] = par.a_grid[i_a]
                c[t,i_a] = par.a_grid[i_a] + par.income_u[t] - par.a_grid[i_a] / (par.R)
            
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
        

                lower_bound = par.L
                upper_bound = (par.a_grid[i_a] + par.income_u[t])*par.R - 10e-6
                upper_bound = min(upper_bound, par.A_0)
                result = minimize_scalar(objective_function_ti, bounds=(lower_bound, upper_bound), args=(par, t, V_u_next[t+1,:]), method='bounded')

                             
                # Extract optimal a_next
                if result.success:
                                      
                    a_next[t, i_a] = result.x
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
                    print("Current function value:", result.fun)
                    print("Optimization result details:", result)

            V_u_next[t,i_a] = V_u[t,i_a]
        
    sol.s = s
    sol.a_next = a_next
    sol.c = c



### Solve forward from initial assets to get true search and consumption path ###

def solve_forward(par, sol, sim):
    s = np.zeros(par.T)
    c = np.zeros(par.T)
    a_next = np.zeros(par.T)
    # b. solve
    for t in range(par.T):
        if t == 0: # First period
            a = par.A_0
            s[t] = sol.s[t,-1]
            a_next[t] = sol.a_next[t,-1]
            c[t] = a + par.income_u[t] - a_next[t] / par.R
        else:
            a = a_next[t-1]
            s_interp = interp1d(par.a_grid, sol.s[t, :])
            a_next_interp = interp1d(par.a_grid, sol.a_next[t, :])
            s[t] = s_interp(a)
            a_next[t] = a_next_interp(a)
            c[t] = a + par.income_u[t] - a_next[t] / par.R
    
    sim.s = s
    sim.a_next = a_next
    sim.c = c


# def solve_forward_employment(par, sol, sim):

#     a_next = np.zeros((par.T, par.N+par.M+1))
#     # b. solve
#     for t in range(par.T):
#         for n in range(par.N+par.M):
#             if t == 0: # First period
#                 if n == 0:
#                     a_next[t, n] = par.A_0

#                 if n == 1:
#                     a = par.A_0
#                     a_next[t, n] = sol.a_next_e[t, n, -1]
#                 else:
#                     a = a_next[t, n-1]
#                     a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :], fill_value="extrapolate")
#                     a_next[t, n] = a_next_interp(a)
#             else:
#                 if n == 0:
#                     a = sim.a_next[t-1]
#                     a_next[t, n] = sim.a_next[t-1]

#                 elif n == 1:
#                     a = sim.a_next[t-1]
#                     a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :], fill_value="extrapolate")
#                     a_next[t, n] = a_next_interp(a)
#                 else:
#                     a = a_next[t, n-1]
#                     a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :], fill_value="extrapolate")
#                     a_next[t, n] = a_next_interp(a)
#     sim.a_e = a_next

def solve_forward_employment(par, sol, sim):

    a_next = np.zeros((par.T, par.N+par.M))
    c = np.zeros((par.T, par.N+par.M))
    # b. solve
    for t in range(par.T):
        for n in range(par.N+par.M):
            if t == 0: # First period
                if n == 0:
                    a_next[t, n] = par.A_0
                    c[t, n] = par.A_0 + par.w - a_next[t, n] / par.R
                if n == 1:
                    a = a_next[t, n-1]
                    a_next[t, n] = sol.a_next_e[t, n, -1]
                    c[t, n] = a + par.w - a_next[t, n] / par.R
                else:
                    a = a_next[t, n-1]
                    a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :], fill_value="extrapolate")
                    a_next[t, n] = a_next_interp(a)
                    c[t, n] = a + par.w - a_next[t, n] / par.R
            else:
                if n == 0:
                    a = sim.a_next[t-1]
                    a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :], fill_value="extrapolate")
                    a_next[t, n] = a_next_interp(a)
                    c[t, n] = a + par.w - a_next[t, n] / par.R

                elif n == 1:
                    a = a_next[t,n-1]
                    a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :], fill_value="extrapolate")
                    a_next[t, n] = a_next_interp(a)
                    c[t, n] = a + par.w - a_next[t, n] / par.R
                else:
                    a = a_next[t, n-1]
                    a_next_interp = interp1d(par.a_grid, sol.a_next_e[t, n, :], fill_value="extrapolate")
                    a_next[t, n] = a_next_interp(a)
                    c[t, n] = a + par.w - a_next[t, n] / par.R
    sim.a_e = a_next
    sim.c_e = c



