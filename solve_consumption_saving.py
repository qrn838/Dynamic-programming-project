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
def objective_function(a_next, par, i_a, i_t, i_n, V_e_next):
    r = par.r_e_m[i_t, i_t + i_n]
    c = par.a_grid[i_a] + par.w - a_next / (par.R)
    penalty = 0
    if c < 1e-6:    # To ensure non-negative consumption (if bounds fail)
        penalty = 1e8*(c)**2

    V_e_next_interp = interp1d(par.a_grid, V_e_next)    # Interpolate V_e_next given a_next
    V_e = utility(par, c, r) + par.delta * V_e_next_interp(a_next) - penalty
    return -V_e

def value_function_employment_VFI(par, sol):
    """Value function when employed"""
    V_e = np.zeros((par.T, par.N + par.M, par.Na))
    a_next = np.zeros((par.T, par.N + par.M, par.Na))
    c = np.zeros((par.T, par.N + par.M, par.Na))

    for i_t in range(par.T):    # Loop over when becoming employed
        for i_n in reversed(range(par.N + par.M)):  # Loop over periods after getting employed, backwards from steady state
            for i_a in range(par.Na):   # Loop over asset grid

                if i_n == par.N + par.M - 1:  # Stationary state
                    r = par.r_e_m[i_t, i_t + i_n]

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
    for t in range(par.T):
        c1 = np.zeros((par.Na, par.N + par.M ))
        c2 = np.zeros((par.Na, par.N + par.M ))
        for n in reversed(range(par.N + par.M )):
     

            # Last period
            if n == par.N + par.M -1:
                sol.c_e[t, n, :] = par.m_grid                   
                sol.a_next_e[t, n, :] = (par.m_grid - sol.c_e[t, n, :])*par.R
                par.V_e[t, n, :] = utility(par, sol.c_e[t, n, :], par.r_e_m[t, t+n])/ (1 - par.delta)
            
            elif n == par.N + par.M -2:
                # Pay off debt
                sol.a_next_e[t, n, :] = 0
                sol.c_e[t, n, :] = par.a_grid + par.w - sol.a_next_e[t, n, :]/par.R
                
                
                # Value function
                V_e_next = interp1d(par.a_grid, par.V_e[t, n+1, :])(sol.a_next_e[t, n, :])
                par.V_e[t, n, :] = utility(par, sol.c_e[t, n, :], par.r_e_m[t, t+n]) + par.delta * V_e_next
            else:
                
                m_temp1 = np.zeros(par.Na+1)
                c_temp1 = np.zeros(par.Na+1)
             
                a1 = np.zeros(par.Na)
                m_temp2 = np.zeros(par.Na+1)
                c_temp2 = np.zeros(par.Na+1)
            
                a2 = np.zeros(par.Na)
                V1 = np.zeros(par.Na)
                V2 = np.zeros(par.Na)

                m_plus = par.a_grid + par.w     # Cash-on-hand for next period
                c_plus = interp1d(par.m_grid, sol.c_e[t, n+1, :], fill_value='extrapolate')(m_plus) # Consumption for next period
                mu_plus = marginal_utility(par, c_plus, par.r_e_m[t, t+n+1])    # Marginal utility for next period

                # Solve above and below ref point
                for i_a in range(par.Na):
                    c_temp1[i_a+1] = inv_marg_utility_1(par, par.delta*par.R*mu_plus[i_a])
                    m_temp1[i_a+1] = c_temp1[i_a+1] + par.a_grid[i_a]/par.R

                    c_temp2[i_a+1] = inv_marg_utility_2(par, par.delta*par.R*mu_plus[i_a])
                    m_temp2[i_a+1] = c_temp2[i_a+1] + par.a_grid[i_a]/par.R

                # Calculate current consumption as function of current cash-on-hand
                c1[:,n] = interp1d(m_temp1, c_temp1, fill_value='extrapolate')(par.m_grid[:]) 
                a1 = (par.m_grid - c1[:,n])*par.R   # Next period assets

                c2[:,n] = interp1d(m_temp2, c_temp2, fill_value='extrapolate')(par.m_grid[:])
                a2 = (par.m_grid - c2[:,n])*par.R   # Next period assets

                # Check if constraints are violated
                for i_a in range(par.Na):
                    if a1[i_a] > par.A_0:
                        c1[i_a,n] = c1[i_a,n] + (a1[i_a] - par.A_0)/par.R
                        a1[i_a] = par.A_0
                    if a1[i_a] < par.L:
                        print("Error: a_next_e out of bounds")
                    
                    if a2[i_a] > par.A_0:
                        c2[i_a,n] = c2[i_a,n] + (a2[i_a] - par.A_0)/par.R
                        a2[i_a] = par.A_0
                    if a2[i_a] < par.L:
                        print("Error: a_next_e out of bounds")

                # Value of employment below kink
                V1 = utility(par, c1[:,n], par.r_e_m[t, t+n]) + par.delta * interp1d(par.a_grid, par.V_e[t, n+1, :], fill_value='extrapolate')(a1)
                # Value of employment above kink
                V2 = utility(par, c2[:,n], par.r_e_m[t, t+n]) + par.delta * interp1d(par.a_grid, par.V_e[t, n+1, :], fill_value='extrapolate')(a2)

                # If no kink use euler else optimizer
                for i_a in range(par.Na):
                    if c1[i_a,n] >= par.r_e_m[t, t+n] and c2[i_a,n] >= par.r_e_m[t, t+n] and c1[i_a,n+1] >= par.r_e_m[t, t+n+1] and c2[i_a,n+1] >= par.r_e_m[t, t+n+1] and not (par.N - 1 <= n <= par.N):
                        sol.c_e[t, n, i_a] = c1[i_a,n]
                        sol.a_next_e[t, n, i_a] = a1[i_a]
                        par.V_e[t, n, i_a] = V1[i_a]
                    elif c1[i_a,n] < par.r_e_m[t, t+n] and c2[i_a,n] < par.r_e_m[t, t+n] and c1[i_a,n+1] < par.r_e_m[t, t+n+1] and c2[i_a,n+1] < par.r_e_m[t, t+n+1] and not (par.N - 1 <= n <= par.N): 
                        sol.c_e[t, n, i_a] = c2[i_a,n]
                        sol.a_next_e[t, n, i_a] = a2[i_a]
                        par.V_e[t, n, i_a] = V2[i_a]
                    else:
                        # Run optimizer
                        lower_bound = par.L
                        upper_bound = (par.a_grid[i_a] + par.w) * par.R - 1e-6  # consumption must be positive
                        upper_bound = min(upper_bound, par.A_0)
                        result = minimize_scalar(objective_function, bounds=(lower_bound, upper_bound), args=(par, i_a, t, n, par.V_e[t, n + 1, :]), method='bounded')
                        if result.success:
                            sol.a_next_e[t, n, i_a] = result.x
                            par.V_e[t, n, i_a] = -result.fun
                            sol.c_e[t, n, i_a] = par.a_grid[i_a] + par.w - sol.a_next_e[t, n, i_a] / par.R
                        
    par.V_e_t_a = par.V_e[:, 0, :]





def unemployment_ss_ConSav(par,t, i_a, i):
    """ Solve for search effort and value function in steady state when unemployed"""
    V_e = par.V_e_t_a[t, i_a]
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
                        V_e_next_interp = interp1d(par.a_grid, par.V_e_t_a[t+1, :])    # Interpolate value of getting employed next period
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
                        V_e_next_interp = interp1d(par.a_grid, par.V_e_t_a[t+1, :])
                        V_e_next = V_e_next_interp(a_next[type, t,i_a])
                        V_u_next_interp = interp1d(par.a_grid, V_u[type,t+1,:])
                        V_u_next = V_u_next_interp(a_next[type,t, i_a])
                        s[type,t,i_a] = inv_marg_cost(par, par.delta*(V_e_next-V_u_next))[type]
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
            s[t] = type_shares @ sim.s[:par.types,t]
        else:
            type_shares = type_shares * (1-sim.s[:par.types,t-1])
            type_shares = type_shares / sum(type_shares)
            s[t] = type_shares @ sim.s[:par.types,t]
    
    sim.s_total[:] = s[:par.T_sim]





########### When employed   ##############
def solve_forward_employment_ConSav(type, par, sol, sim):

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
                    a = sim.a_next[type,t-1]
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



