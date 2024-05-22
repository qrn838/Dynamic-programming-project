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


def value_function_employment_ConSav(par, sol):
    '''Value function using either VFI or EGM for the employment state'''
    if par.euler == False:
        value_function_employment_VFI(par, sol)
    else:
        value_function_employment_EGM(par, sol)


########## VFI #########
def objective_function_ConSav(a_next, par, i_a, i_t, i_n, V_e_next):
    r = par.r_e_m[i_t, i_t + i_n]                       # Reference point
    c = par.a_grid[i_a] + par.w - a_next / (par.R)      # Consumption
    V_e_next_interp = interp1d(par.a_grid, V_e_next)    # Interpolate value function for next period
    V_e = utility(par, c, r) + par.delta * V_e_next_interp(a_next)  # Current value function
    return -V_e

def value_function_employment_VFI(par, sol):
    """Value function when employed using VFI"""
    V_e = np.zeros((par.types,par.T, par.N + par.M, par.Na))
    a_next = np.zeros((par.types,par.T, par.N + par.M, par.Na))
    c = np.zeros((par.types,par.T, par.N + par.M, par.Na))

    for type in range(par.types):       # Loop over types
        for i_t in range(par.T):        # Loop over periods of getting employed
            for i_n in range(par.N + par.M - 1, -1, -1):    # Loop over time after employment
                for i_a in range(par.Na):                   # Loop over asset grid

                    if i_n == par.N + par.M - 1:            # Last period stationary state
                        r = par.r_e_m[i_t, i_t + i_n]
                        c[type,i_t, i_n, i_a] = par.a_grid[i_a] + par.w - par.a_grid[i_a] / par.R
                        V_e[type,i_t, i_n, i_a] = utility(par, c[type,i_t, i_n, i_a], r) / (1 - par.delta)  
                        a_next[type,i_t, i_n, i_a] = par.a_grid[i_a]

                    
                    else:  # Consumption saving problem choosing a_next
                        # Bounds for the optimizer
                        lower_bound = par.L
                        upper_bound = (par.a_grid[i_a] + par.w) * par.R - 1e-6  # consumption must be positive
                        upper_bound = min(upper_bound, par.A_0)

                        # Run the optimizer using minimize_scalar with method='golden'
                        result = minimize_scalar(objective_function_ConSav, bounds=(lower_bound, upper_bound), args=(par, i_a, i_t, i_n, V_e[type,i_t, i_n + 1, :]), method='bounded')

                        if result.success:
                            a_next[type,i_t, i_n, i_a] = result.x
                            V_e[type,i_t, i_n, i_a] = -result.fun
                            c[type,i_t, i_n, i_a] = par.a_grid[i_a] + par.w - a_next[type,i_t, i_n, i_a] / par.R
                        else:
                            print(f"Error at t={i_t}, i_n={i_n}, i_a={i_a}")
                            print("Error message:", result.message)
                            print("Current function value:", result.fun)
                            print("Optimization result details:", result)

    # Store results in containers
    par.V_e_t_a = V_e[:, :, 0, :]
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

    # for i_t in range(par.T): 
    #     for i_n in range(par.N + par.M - 1, -1, -1):
    for i_t in range(par.T):
        for i_a in range(par.Na):
            r = par.r_e_m[i_t, -1]
            c = par.a_grid[i_a] + par.w - par.a_grid[i_a] / par.R
            V_e[i_t, -1, i_a] = utility(par, c, r) / (1 - par.delta)  ## stationary state
            a_next[i_t, -1, i_a] = par.a_grid[i_a]

            a_egm[i_t, -1, i_a] = par.a_grid[i_a]/par.R + c - par.w
            # m_egm[i_t, i_n, i_a] = par.a_grid[i_a] + par.w
            c_egm[i_t, -1, i_a] = par.a_grid[i_a] + par.w - par.a_grid[i_a] / par.R

        
    for i_t in range(par.T):
        for i_n in range(par.N + par.M - 2, -1, -1):
            a = par.a_grid  # Exogenous grid
            c_next_int = interp1d(a_egm[i_t, i_n+1,:],c_egm[i_t, i_n+1,:], fill_value='extrapolate')
            c_next = c_next_int(par.a_grid)

            marg_util_next = marginal_utility(par, c_next, par.r_e_m[i_t, i_t + i_n + 1])
            c1 = inv_marg_utility_1(par, par.delta * par.R *marg_util_next)
            c2 = inv_marg_utility_2(par, par.delta * par.R *marg_util_next)

            a1 = par.a_grid[:] / par.R + c1 - par.w  # Cash on hand needed this period to obtain c and next period assets
            a2 = par.a_grid[:] / par.R + c2 - par.w

            #next period assets as function of current period assets
            a1_next = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a1)
            a2_next = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a2)
        
            for i in range(par.Na):
                if a1[i] > par.A_0:
                    a1[i] = par.A_0
                    c1[i] = a1[i] + par.w - par.a_grid[i] / par.R
                    a1_next[i] = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a1[i])

                if a2[i] > par.A_0:
                    a2[i] = par.A_0
                    c2[i] = a2[i] + par.w - par.a_grid[i] / par.R
                    a2_next[i] = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a2[i])
                    
                if a1[i] < par.L:
                    a1[i] = par.L
                    c1[i] = a1[i] + par.w - par.a_grid[i] / par.R
                    a1_next[i] = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a1[i])
                if a2[i] < par.L:
                    a2[i] = par.L
                    c2[i] = a2[i] + par.w - par.a_grid[i] / par.R
                    a2_next[i] = interp1d(par.a_grid, a_egm[i_t, i_n + 1, :], fill_value="extrapolate")(a2[i])


            # a1_next = interp1d(a1, par.a_grid, fill_value="extrapolate")(par.a_grid)
            # a2_next = interp1d(a2, par.a_grid, fill_value="extrapolate")(par.a_grid)
            # print(a1_next)

            for i in range(par.Na):
                
                if c1[i] >= par.r_e_m[i_t, i_t + i_n] and c2[i] >= par.r_e_m[i_t, i_t + i_n]:
                    c_egm[i_t, i_n, i] = c1[i]
                    a_egm[i_t, i_n, i] = a1_next[i]
                    # a_egm[i_t, i_n, i] = a1[i]
                    
                    
                elif  c1[i] < par.r_e_m[i_t, i_t + i_n] and c2[i] < par.r_e_m[i_t, i_t + i_n]:
                    c_egm[i_t, i_n, i] = c2[i]
                    a_egm[i_t, i_n, i] = a2_next[i]
                    # a_egm[i_t, i_n, i] = a2[i]
                    
            
                else:
                    print('Error in EGM at time period', i_t, 'and state', i_n, 'and asset', i)
                    V_e_1 = utility(par, c1[i], par.r_e_m[i_t, i_t + i_n]) + par.delta * interp1d(par.a_grid, V_e[i_t, i_n + 1, :], fill_value="extrapolate")(a1[i])
                    V_e_2 = utility(par, c2[i], par.r_e_m[i_t, i_t + i_n]) + par.delta * interp1d(par.a_grid, V_e[i_t, i_n + 1, :], fill_value="extrapolate")(a2[i])

                    if V_e_1 > V_e_2:
                        c_egm[i_t, i_n, i] = c1[i]
                        a_egm[i_t, i_n, i] = a1[i]
                        # a_next[i_t, i_n, i] = a1_next[i]
                    
                    else:
                        c_egm[i_t, i_n, i] = c2[i] 
                        a_egm[i_t, i_n, i] = a2[i]
                        # a_next[i_t, i_n, i]= a2_next[i]
                        
                

                V_e[i_t, i_n, i] = utility(par, c_egm[i_t, i_n, i], par.r_e_m[i_t, i_t + i_n]) + par.delta * interp1d(par.a_grid, V_e[i_t, i_n + 1, :], fill_value="extrapolate")(a_egm[i_t, i_n, i])
                a_next[i_t, i_n, i] = a_egm[i_t, i_n, i]
                    # Debug prints
                # print(f"t={i_t}, i_n={i_n}, i_a={i_a}")
                # print(f"a_grid={par.a_grid[i_a]}")
                # print(f"c1={c1}, c2={c2}")
                # print(f"m1={m1}, m2={m2}")
                # print(f"a1={a1}, a2={a2}")
                # print(f"c_egm={c_egm[i_t, i_n, i_a]}, a_next={a_next[i_t, i_n, i_a]}")
                # print(f"V_e={V_e[i_t, i_n, i_a]}")


    par.V_e_t_a = V_e[:, 0, :]
    par.V_e = V_e
    sol.a_next_e = a_egm
    sol.c_e = c_egm



# def unemployment_ss_ConSav(par, t, i_a, type):
#     '''Find the stationary state of the unemployment value function'''
#     V_e = par.V_e_t_a[type, t, i_a]
#     c = par.a_grid[i_a] + par.income_u[t] - par.a_grid[i_a] / (par.R)
#     r = par.r_u[t]

#     def bellman_difference(V_u):
#         '''Objective function for solver'''
#         s = inv_marg_cost(par, par.delta * (V_e - V_u))[type]
#         V_u_new = utility(par, c, r) - cost(par, s)[type] + par.delta * (s * V_e + (1 - s) * V_u)
#         return V_u_new - V_u

#     # Initial values for V_u
#     a, b = -5, 5
#     fa = bellman_difference(a)
#     fb = bellman_difference(b)

#     # If no root: Try different starting values
#     if fa * fb > 0:
#         interval_found = False

#         factors = [-500, -300, -100, -50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50, 100, 200, 500]
#         for factor_a in factors:
#             for factor_b in factors:
#                 fa = bellman_difference(factor_a)
#                 fb = bellman_difference(factor_b)
#                 if fa * fb < 0:
#                     a, b = factor_a, factor_b
#                     interval_found = True
#                     break
#         if not interval_found:
#             raise ValueError("Could not find a valid interval where the function has different signs.")

#     # Save results
#     V_u = brentq(bellman_difference, a, b)
#     s = inv_marg_cost(par, par.delta * (V_e - V_u))[type]

#     return V_u, s

def unemployment_ss_ConSav(par,t, i_a, i):
    """ Solve for search effort and value function in steady state when unemployed"""
    V_e = par.V_e_t_a[i, t, i_a]
    c = par.a_grid[i_a] + par.income_u[t] - par.a_grid[i_a] / (par.R)
    r = par.r_u[t]

    def objective_function(s, par, i, V_e, c, r):
        V_u = (utility(par, c, r) - cost(par,s)[i] + par.delta * (s * V_e)) / (1-par.delta*(1-s))
        return -V_u  # Minimize the negative of V_u to maximize V_u

    # Perform optimization
    s_initial_guess = 0.8
    result = minimize(objective_function, s_initial_guess, args=(par,i,V_e, c, r), bounds=[(0, 1)])

    # Extract optimal s
    optimal_s_ss = result.x[0]
    V_u_ss = -result.fun

    return optimal_s_ss, V_u_ss





### Backward Induction to solve search effort in all periods of unemployment ###

def solve_search_and_consumption_ConSav(par, sol):
    '''Solve the search and consumption problem for all periods of unemployment using backward induction'''
    
    # Containers
    tuple = (par.types, par.T, par.Na)
    s = np.zeros(tuple)
    V_u = np.zeros(tuple)
    c = np.zeros(tuple)
    a_next = np.zeros(tuple)

    # Backwards iteration
    for type in range(par.types):               # Loop over types
        for t in range(par.T - 1, -1, -1):      # Loop backwards over periods
            for i_a in range(par.Na):           # Loop over asset grid
                if t == par.T - 1:              # Stationary state
                    V_u[type,t,i_a] = unemployment_ss_ConSav(par, t, i_a, type)[1]
                    s[type,t,i_a] = unemployment_ss_ConSav(par, t, i_a, type)[0]
                    a_next[type,t,i_a] = par.a_grid[i_a]
                    c[type,t,i_a] = par.a_grid[i_a] + par.income_u[t] - par.a_grid[i_a] / (par.R)
                
                else: # Consumption saving problem choosing a_next
                    def objective_function_ti(a_next, par,t,V_u_next, type):
                        '''Objective function for solver'''
                        income = par.income_u[t]
                        r = par.r_u[t]
                        c = par.a_grid[i_a] + income - a_next / (par.R)
                        V_e_next_interp = interp1d(par.a_grid, par.V_e_t_a[type,t+1, :])    # Interpolate value of getting employed next period
                        V_e_next = V_e_next_interp(a_next)                                  # Value of getting employed next period
                        V_u_next_interp = interp1d(par.a_grid, V_u_next)                    # Interpolate value of unemployment next period
                        V_u_next = V_u_next_interp(a_next)                                  # Value of unemployment next period
                        s = inv_marg_cost(par, par.delta*(V_e_next-V_u_next))[type]         # Search effort
                        # if s > 1:       # Check if search effort is within bounds
                        #     print('obj.s')
                        #     print(s)
                        V_u = utility(par,c,r) - cost(par,s)[type] + par.delta * (s * V_e_next+(1-s)*V_u_next)
                        return -V_u
            

                    # Bounds for savings (used in optimization)
                    lower_bound = par.L         
                    upper_bound = (par.a_grid[i_a] + par.income_u[t])*par.R - 10e-6    
                    upper_bound = min(upper_bound, par.A_0)
                    # Call optimizer
                    result = minimize_scalar(objective_function_ti, bounds=(lower_bound, upper_bound), args=(par, t, V_u[type, t+1,:], type), method='bounded')
                                
                    # Extract optimal a_next
                    if result.success:
                                        
                        a_next[type, t, i_a] = result.x
                        V_u[type, t,i_a] = -result.fun
                        
                        # Get search effort and consumption
                        V_e_next_interp = interp1d(par.a_grid, par.V_e_t_a[type,t+1, :])
                        V_e_next = V_e_next_interp(a_next[type, t,i_a])
                        V_u_next_interp = interp1d(par.a_grid, V_u[type,t+1,:])
                        V_u_next = V_u_next_interp(a_next[type,t, i_a])
                        s[type,t,i_a] = inv_marg_cost(par, par.delta*(V_e_next-V_u_next))[type]
                        if s[type,t,i_a] > 1:
                            print('sol.s')
                            print(s[type,t,i_a])
                        c[type,t,i_a] = par.a_grid[i_a] + par.income_u[t] - a_next[type,t,i_a] / (par.R)
                        
                
                    else:
                        print("Error at t={}, i_a={}".format(t, i_a))
                        print("Error message:", result.message)
                        print("Current function value:", result.fun)
                        print("Optimization result details:", result)
        
    sol.s = s
    sol.a_next = a_next
    sol.c = c



def solve_forward_ConSav(par, sol, sim, type):
    '''Simulate unemployment search path for each type'''

    # Containers
    s = np.zeros(par.T)
    c = np.zeros(par.T)
    a_next = np.zeros(par.T)

    # b. solve
    for t in range(par.T):  # Loop over periods
        if t == 0:          # First period
            a = par.A_0     # Initial assets when getting unemployed
            s[t] = sol.s[type,t,-1]   # Search effort
            a_next[t] = sol.a_next[type,t,-1]   # Next period assets
            c[t] = a + par.income_u[t] - a_next[t] / par.R  # Consumption
        else:
            a = a_next[t-1] # Assets from last period
            s_interp = interp1d(par.a_grid, sol.s[type,t, :])   # Search effort given assets
            a_next_interp = interp1d(par.a_grid, sol.a_next[type,t, :]) # Next period assets given current assets
            s[t] = s_interp(a)  # Search effort
            a_next[t] = a_next_interp(a)    # Next period assets
            c[t] = a + par.income_u[t] - a_next[t] / par.R  # Consumption
    
    sim.s[type,:] = s
    sim.a_next[type,:] = a_next
    sim.c[type,:] = c




def sim_search_effort_ConSav(par, sol, sim):
    '''Simulate combined search effort across all types and periods for consumption saving model'''

    # Container
    s = np.zeros(par.T)

    # Relevant types
    type_shares = [par.type_shares1, par.type_shares2, par.type_shares3]    
    type_shares = type_shares[:par.types]

    # Solve forward for each type
    for i in range(par.types):
        solve_forward_ConSav(par, sol, sim, i)

    # Calculate total search effort as weighted average over types
    for t in range(par.T):
        if t == 0:
            s[t] = type_shares @ sim.s[:,t]
        else:
            type_shares = type_shares * (1-sim.s[:,t-1])
            type_shares = type_shares / sum(type_shares)
            s[t] = type_shares @ sim.s[:,t]
    
    sim.s_total[:] = s[:par.T_sim]





###########  Not made with types yet  ##############
def solve_forward_employment_ConSav(par, sol, sim):

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



