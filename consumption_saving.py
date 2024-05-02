import numpy as np
import copy
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from Funcs import *



def value_function_employment(par):
    """Value function when employed"""
    V_e = np.zeros((par.T, par.N+par.M, par.Na))
    V_e_next = np.zeros((par.T, par.N+par.M, par.Na))   # M to ensure that we have converged to stationary state. Check this is the case
    a_next = np.zeros((par.T, par.N+par.M, par.Na))

    for i_t in range(par.T): 
        for i_n in range(par.N+par.M-1, -1, -1):
            for i_a in range(par.Na):

                if i_n == par.N+par.M - 1:  # Stationary state
                    V_e[i_t, i_n, i_a] = consumption_utility(par.w) / (1 - par.delta)  ## stationary state
                    
                elif i_n == par.N+par.M - 2: # Last period before stationary state. Repay remaining debt
                    r = par.r_e_m[i_t, i_t + i_n]
                    c = par.a_grid[i_a] + par.w
                    V_e[i_t, i_n, i_a] = utility(par, c, r) + par.delta * V_e_next[i_t, i_n+1, i_a]
                    a_next[i_t, i_n, i_a] = 0.0

                else: # Cosumption saving problem
                    if par.euler == False:  # With optimizer
                        def objective_function(a_next, par, i_a, i_t, i_n, V_e_next):
                            r = par.r_e_m[i_t, i_t + i_n]
                            c = par.a_grid[i_a] + par.w - a_next / (par.R)
                            V_e_next_interp = interp1d(par.a_grid, V_e_next)
                            V_e = utility(par, c, r) + par.delta * V_e_next_interp(a_next)
                            return -V_e

                        a_next_guess = par.a_grid[i_a]
                        result = minimize(objective_function, a_next_guess, args=(par, i_a, i_t, i_n, V_e_next[i_t, i_n+1, :]),
                                        bounds=[(par.L, par.A_0)])
                        a_next[i_t, i_n, i_a] = result.x[0]
                        V_e[i_t, i_n, i_a] = -result.fun

                    elif par.euler == True:  # Using euler equation. Figure out how to do this
                    
                        r = par.r_e_m[i_t, i_t + i_n]
                        r_next = par.r_e_m[i_t, i_t + i_n + 1]
                        c_next = par.a_grid[i_a] + par.w - a_next[i_t, i_n+1, i_a] / (par.R)
                        
           
                        print(r_next)
                        print(c_next)

                        x = par.delta*par.R*marginal_utility(par,c_next,r_next)  # Use euler here to get consumption given next period. Is this done right??
                        c1 = inv_marg_utility_1(par,x)
                        c2 = inv_marg_utility_2(par,x)
                        
                        if c1 >= r and c2 >= r: # If both are above are use euler where c>=r
                            c = c1
                            a_next[i_t, i_n, i_a] = (par.a_grid[i_a] + par.w - c) * (par.R)
                            
                            if a_next[i_t, i_n, i_a]<par.L:  #Enforce borrowing constraint. Is this done right?
                                a_next[i_t, i_n, i_a] = par.L
                                c = par.a_grid[i_a] + par.w - a_next[i_t, i_n, i_a] / (par.R)
                            V_e_next_interp = interp1d(par.a_grid,  V_e_next[i_t, i_n+1, :])
                            V_e[i_t, i_n, i_a] = utility(par, c, r) + par.delta *V_e_next_interp(a_next[i_t, i_n, i_a]) 
                            
                        elif c1 < r and c2 < r: # If both are below r use euler where c<r
                            c = c2
                            a_next[i_t, i_n, i_a] = (par.a_grid[i_a] + par.w - c) * (par.R)
                            if a_next[i_t, i_n, i_a]<par.L:  #Enforce borrowing constraint. Is this done right?
                                a_next[i_t, i_n, i_a] = par.L
                                c = par.a_grid[i_a] + par.w - a_next[i_t, i_n, i_a] / (par.R)
                            V_e_next_interp = interp1d(par.a_grid,  V_e_next[i_t, i_n+1, :])
                            V_e[i_t, i_n, i_a] = utility(par, c, r) + par.delta *V_e_next_interp(a_next[i_t, i_n, i_a]) 
                
                        else:  # If one is above and one is below r use numerical optimizer
                            def objective_function(a_next, par, i_a, i_t, i_n, V_e_next):
                                r = par.r_e_m[i_t, i_t + i_n]
                                c = par.a_grid[i_a] + par.w - a_next / (par.R)
                                V_e_next_interp = interp1d(par.a_grid,  V_e_next)
                                V_e = utility(par, c, r) + par.delta * V_e_next_interp(a_next)
                                return -V_e

                            a_next_guess = par.a_grid[i_a]
                            result = minimize(objective_function, a_next_guess, args=(par, i_a, i_t, i_n, V_e_next[i_t, i_n+1, :]),
                                            bounds=[(par.L, par.A_0)])
                            a_next[i_t, i_n, i_a] = result.x[0]
                            V_e[i_t, i_n, i_a] = -result.fun
                            
     
                V_e_next[i_t, i_n, i_a] = V_e[i_t, i_n, i_a]
                
        
    par.V_e_t_a = V_e[:, 0, :]






### Backward Induction to solve search effort in all periods of unemployment ###

def solve_search_and_consumption(par):
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
            if t == par.T - 1:
                def objective_function(s, par):
                    V_e = par.V_e_t_a[par.T - 1,0]
                    c = par.b3+(par.R-1)*par.L
                    V_u = (consumption_utility(c) - cost(s) + par.delta * (s * V_e)) / (1-par.delta*(1-s))
                    return -V_u
                
                s_initial_guess = 0.8
                # Perform optimization
                result = minimize(objective_function, s_initial_guess, args=(par,), method='Nelder-Mead', bounds=[(0, 1)])
                if result.success:
                    # Extract optimal s
                    s[t,i_a] = result.x[0]
                    V_u[t,i_a] = -result.fun
                    c[t,i_a] = par.b3+(par.R-1)*par.L
                    a_next[t,i_a] = par.L
                else:
                    print("Optimization failed at t={}, i_a={}".format(t, i_a))
            
            else:
                def objective_function(a_next, par,t,V_u_next):
                    income = par.income_u[t]
                    r = par.r_u[t]
                    c = par.a_grid[i_a] + income - a_next / (par.R)
                    V_e_next_interp = interp1d(par.a_grid, par.V_e_t_a[t+1, :])
                    V_e_next = V_e_next_interp(a_next)
                    s = inv_marg_cost(par.delta*(V_e_next-V_u_next))
                    V_u = (utility(par,c,r) - cost(s) + par.delta * (s * V_e_next+(1-s)*V_u_next))
                    return -V_u
                
                a_next_guess = par.a_grid[i_a]
                # Perform optimization

                result = minimize(objective_function, a_next_guess, args=(par, t, V_u_next[t+1,i_a]), method='Nelder-Mead', bounds=[(par.L, par.A_0)])
                             
                # Extract optimal a_next
                if result.success:
                    
                    a_next[t, i_a] = result.x[0]
                    V_u[t, i_a] = -result.fun
                    V_e_next_interp = interp1d(par.a_grid, par.V_e_t_a[t+1, :])
                    V_e_next = V_e_next_interp(a_next[t,i_a])
                    s[t,i_a] = inv_marg_cost(par.delta*(V_e_next-V_u_next[t+1,i_a]))
                    c[t,i_a] = par.a_grid[i_a] + par.income_u[t] - a_next[t,i_a] / (par.R)
                else:
                    print("Optimization failed at t={}, i_a={}".format(t, i_a))


            
            V_u_next[t,i_a] = V_u[t,i_a]
        

    return s, c, V_u


### Solve forward from initial assets to get true search and consumption path ###
   