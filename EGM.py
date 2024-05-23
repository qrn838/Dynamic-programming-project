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

        


def EGM(par, sol):
    for t in range(par.T):
        for n in reversed(range(par.N + par.M )):
     

            # Last period
            if n == par.N + par.M -1:
                sol.c_e[t, n, :] = par.m_grid                   
                sol.a_next_e[t, n, :] = (par.m_grid - sol.c_e[t, n, :])*par.R
                # sol.c_e[t, n, :] = par.a_grid + par.w - par.a_grid/par.R
                # sol.a_next_e[t, n, :] = par.a_grid
                par.V_e[t, n, :] = utility(par, sol.c_e[t, n, :], par.r_e_m[t, t+n])/ (1 - par.delta)
                # print(par.V_e[t, n, :])
            elif n == par.N + par.M -2:
                sol.a_next_e[t, n, :] = 0
                sol.c_e[t, n, :] = par.a_grid + par.w - sol.a_next_e[t, n, :]/par.R
                V_e_next = interp1d(par.a_grid, par.V_e[t, n+1, :])(sol.a_next_e[t, n, :])
                par.V_e[t, n, :] = utility(par, sol.c_e[t, n, :], par.r_e_m[t, t+n]) + par.delta * V_e_next
            else:
                m_temp = np.zeros(par.Na+1)
                c_temp = np.zeros(par.Na+1)
                for i_a in range(par.Na):
                    m_plus = par.a_grid[i_a] + par.w
                    c_plus = interp1d(par.m_grid, sol.c_e[t, n+1, :], fill_value='extrapolate')(m_plus)
                    mu_plus = marginal_utility(par, c_plus, par.r_e_m[t, t+n+1])

                    c_temp[i_a+1] = inv_marg_utility_1(par, par.delta*par.R*mu_plus)
                    m_temp[i_a+1] = c_temp[i_a+1] + par.a_grid[i_a]/par.R

                sol.c_e[t, n, :] = interp1d(m_temp, c_temp, fill_value='extrapolate')(par.m_grid[:])
                sol.a_next_e[t, n, :] = (par.m_grid - sol.c_e[t, n, :])*par.R
                for i_a in range(par.Na):
                    if sol.a_next_e[t, n, i_a] > par.A_0:
                        sol.c_e[t, n, i_a] = sol.c_e[t, n, i_a] + (sol.a_next_e[t, n, i_a] - par.A_0)/par.R
                        sol.a_next_e[t, n, i_a] = par.A_0
                        
                    if sol.a_next_e[t, n, i_a] < par.L:
                        print("Error: a_next_e out of bounds")

                
                V_e_next = interp1d(par.a_grid, par.V_e[t, n+1, :])(sol.a_next_e[t, n, :])
                par.V_e[t, n, :] = utility(par, sol.c_e[t, n, :], par.r_e_m[t, t+n]) + par.delta * V_e_next
                # print(par.V_e[t, n, :])
            

