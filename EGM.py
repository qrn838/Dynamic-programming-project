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


# def EGM(par, sol):
#     for t in range(par.T):
#         for n in reversed(range(par.N + par.M )):
     

#             # Last period
#             if n == par.N + par.M -1:
#                 sol.c_e[t, n, :] = par.m_grid                   
#                 sol.a_next_e[t, n, :] = (par.m_grid - sol.c_e[t, n, :])*par.R
#                 par.V_e[t, n, :] = utility(par, sol.c_e[t, n, :], par.r_e_m[t, t+n])/ (1 - par.delta)
            
#             elif n == par.N + par.M -2:
#                 # Pay off debt
#                 sol.a_next_e[t, n, :] = 0
#                 sol.c_e[t, n, :] = par.a_grid + par.w - sol.a_next_e[t, n, :]/par.R

#                 # Handle case with negative consumption
#                 # sol.a_next_e[t, n, :] = np.where(sol.c_e[t, n, :] <= 0, (par.m_grid - 1e-6)*par.R, sol.a_next_e[t, n, :])
#                 # sol.c_e[t, n, :] = np.where(sol.c_e[t, n, :] <= 0, 1e-6, sol.c_e[t, n, :])
                
#                 # Value function
#                 V_e_next = interp1d(par.a_grid, par.V_e[t, n+1, :])(sol.a_next_e[t, n, :])
#                 par.V_e[t, n, :] = utility(par, sol.c_e[t, n, :], par.r_e_m[t, t+n]) + par.delta * V_e_next
#             else:
#                 m_temp1 = np.zeros(par.Na+1)
#                 c_temp1 = np.zeros(par.Na+1)
#                 m_temp2 = np.zeros(par.Na+1)
#                 c_temp2 = np.zeros(par.Na+1)
#                 a1 = np.zeros(par.Na)
#                 a2 = np.zeros(par.Na)
#                 c1 = np.zeros(par.Na)
#                 c2 = np.zeros(par.Na)
#                 V1 = np.zeros(par.Na)
#                 V2 = np.zeros(par.Na)

#                 # Solve above reference point
#                 for i_a in range(par.Na):
#                     m_plus = par.a_grid[i_a] + par.w
#                     c_plus = interp1d(par.m_grid, sol.c_e[t, n+1, :], fill_value='extrapolate')(m_plus)
#                     mu_plus = marginal_utility(par, c_plus, par.r_e_m[t, t+n+1])

#                     c_temp1[i_a+1] = inv_marg_utility_1(par, par.delta*par.R*mu_plus)
#                     m_temp1[i_a+1] = c_temp1[i_a+1] + par.a_grid[i_a]/par.R

#                     c1[i_a] = interp1d(m_temp1, c_temp1, fill_value='extrapolate')(par.m_grid[i_a])
#                     a1[i_a] = (par.m_grid[i_a] - c1[i_a])*par.R
#                     if a1[i_a] > par.A_0:
#                         c1[i_a] = c1[i_a] + (a1[i_a] - par.A_0)/par.R
#                         a1[i_a] = par.A_0
#                     if a1[i_a] < par.L:
#                         c1[i_a] = c1[i_a] + (a1[i_a] - par.L )/par.R
#                         a1[i_a] = par.L   
#                 V1 = utility(par, c1, par.r_e_m[t, t+n+1]) + par.delta*interp1d(par.a_grid, par.V_e[t, n+1, :])(a1)

#                 #Solve below reference point
#                 for i_a in range(par.Na):
#                     m_plus = par.a_grid[i_a] + par.w
#                     c_plus = interp1d(par.m_grid, sol.c_e[t, n+1, :], fill_value='extrapolate')(m_plus)
#                     mu_plus = marginal_utility(par, c_plus, par.r_e_m[t, t+n+1])

#                     c_temp2[i_a+1] = inv_marg_utility_2(par, par.delta*par.R*mu_plus)
#                     m_temp2[i_a+1] = c_temp2[i_a+1] + par.a_grid[i_a]/par.R

#                     c2[i_a] = interp1d(m_temp2, c_temp2, fill_value='extrapolate')(par.m_grid[i_a])
#                     a2[i_a] = (par.m_grid[i_a] - c2[i_a])*par.R
#                     if a2[i_a] > par.A_0:
#                         c2[i_a] = c2[i_a] + (a2[i_a] - par.A_0)/par.R
#                         a2[i_a] = par.A_0
#                     if a2[i_a] < par.L:
#                         c2[i_a] = c2[i_a] + (a2[i_a] - par.L )/par.R
#                         a2[i_a] = par.L
#                 V2 = utility(par, c2, par.r_e_m[t, t+n+1]) + par.delta*interp1d(par.a_grid, par.V_e[t, n+1, :])(a2)

#                 for i_a in range(par.Na):
#                     if c1[i_a] > par.r_e_m[t, t+n+1] and c2[i_a] > par.r_e_m[t, t+n+1]:
#                         sol.c_e[t, n, i_a] = c1[i_a]
#                         sol.a_next_e[t, n, i_a] = a1[i_a]
#                     elif c1[i_a] < par.r_e_m[t, t+n+1] and c2[i_a] < par.r_e_m[t, t+n+1]:
#                         sol.c_e[t, n, i_a] = c2[i_a]
#                         sol.a_next_e[t, n, i_a] = a2[i_a]
#                     else:
#                         if V1[i_a]>=V2[i_a]:
#                             sol.c_e[t, n, i_a] = c1[i_a]
#                             sol.a_next_e[t, n, i_a] = a1[i_a]
#                         else:
#                             sol.c_e[t, n, i_a] = c2[i_a]
#                             sol.a_next_e[t, n, i_a] = a2[i_a]
#                 V_e_next = interp1d(par.a_grid, par.V_e[t, n+1, :])(sol.a_next_e[t, n, :])
#                 par.V_e[t, n, :] = utility(par, sol.c_e[t, n, :], par.r_e_m[t, t+n]) + par.delta * V_e_next




def EGM(par, sol):
    for t in range(par.T):
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
                
                # Handle case with negative consumption
                # sol.a_next_e[t, n, :] = np.where(sol.c_e[t, n, :] <= 0, (par.m_grid - 1e-6)*par.R, sol.a_next_e[t, n, :])
                # sol.c_e[t, n, :] = np.where(sol.c_e[t, n, :] <= 0, 1e-6, sol.c_e[t, n, :])
                
                # Value function
                V_e_next = interp1d(par.a_grid, par.V_e[t, n+1, :])(sol.a_next_e[t, n, :])
                par.V_e[t, n, :] = utility(par, sol.c_e[t, n, :], par.r_e_m[t, t+n]) + par.delta * V_e_next
            else:

                m_temp1 = np.zeros(par.Na+1)
                c_temp1 = np.zeros(par.Na+1)
                c1 = np.zeros(par.Na)
                a1 = np.zeros(par.Na)
                m_temp2 = np.zeros(par.Na+1)
                c_temp2 = np.zeros(par.Na+1)
                c2 = np.zeros(par.Na)
                a2 = np.zeros(par.Na)
                V1 = np.zeros(par.Na)
                V2 = np.zeros(par.Na)

                m_plus = par.a_grid + par.w
                c_plus = interp1d(par.m_grid, sol.c_e[t, n+1, :], fill_value='extrapolate')(m_plus)
                mu_plus = marginal_utility(par, c_plus, par.r_e_m[t, t+n+1])

                # Solve above and below ref point
                for i_a in range(par.Na):
                    c_temp1[i_a+1] = inv_marg_utility_1(par, par.delta*par.R*mu_plus[i_a])
                    m_temp1[i_a+1] = c_temp1[i_a+1] + par.a_grid[i_a]/par.R

                    c_temp2[i_a+1] = inv_marg_utility_2(par, par.delta*par.R*mu_plus[i_a])
                    m_temp2[i_a+1] = c_temp2[i_a+1] + par.a_grid[i_a]/par.R

                c1 = interp1d(m_temp1, c_temp1, fill_value='extrapolate')(par.m_grid[:])
                a1 = (par.m_grid - c1)*par.R

                c2 = interp1d(m_temp2, c_temp2, fill_value='extrapolate')(par.m_grid[:])
                a2 = (par.m_grid - c2)*par.R

                print(a1)
                for i_a in range(par.Na):
                    if a1[i_a] > par.A_0:
                        c1[i_a] = c1[i_a] + (a1[i_a] - par.A_0)/par.R
                        a1[i_a] = par.A_0
                    if a1[i_a] < par.L:
                        print("Error: a_next_e out of bounds")
                    
                    if a2[i_a] > par.A_0:
                        c2[i_a] = c2[i_a] + (a2[i_a] - par.A_0)/par.R
                        a2[i_a] = par.A_0
                    if a2[i_a] < par.L:
                        print("Error: a_next_e out of bounds")

                V1 = utility(par, c1, par.r_e_m[t, t+n+1]) + par.delta * interp1d(par.a_grid, par.V_e[t, n+1, :])(a1)
                V2 = utility(par, c2, par.r_e_m[t, t+n+1]) + par.delta * interp1d(par.a_grid, par.V_e[t, n+1, :])(a2)

                # If always above use 1, if always below use 2, else use highest value function
                for i_a in range(par.Na):
                    if c1[i_a] >= par.r_e_m[t, t+n+1] and c2[i_a] >= par.r_e_m[t, t+n+1]:
                        sol.c_e[t, n, i_a] = c1[i_a]
                        sol.a_next_e[t, n, i_a] = a1[i_a]
                        par.V_e[t, n, i_a] = V1[i_a]
                    elif c1[i_a] < par.r_e_m[t, t+n+1] and c2[i_a] < par.r_e_m[t, t+n+1]:
                        sol.c_e[t, n, i_a] = c2[i_a]
                        sol.a_next_e[t, n, i_a] = a2[i_a]
                        par.V_e[t, n, i_a] = V2[i_a]
                    else:
                        if V1[i_a]>=V2[i_a]:
                            sol.c_e[t, n, i_a] = c1[i_a]
                            sol.a_next_e[t, n, i_a] = a1[i_a]
                            par.V_e[t, n, i_a] = V1[i_a]
                        else:
                            sol.c_e[t, n, i_a] = c2[i_a]
                            sol.a_next_e[t, n, i_a] = a2[i_a]
                            par.V_e[t, n, i_a] = V2[i_a]
                
                # par.V_e[t, n, :] = V2
                # # par.V_e[t, n, :] = utility(par, c2, par.r_e_m[t, t+n]) + par.delta * V_e_next
                # sol.c_e[t, n, :] = c2
                # sol.a_next_e[t, n, :] = a2


   