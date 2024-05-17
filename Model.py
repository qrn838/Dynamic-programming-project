from types import SimpleNamespace
import numpy as np
import scipy
from copy import deepcopy
from scipy.io import loadmat

from EconModel import EconModelClass
from consumption_saving import *


class ReferenceDependenceClass(EconModelClass):

	def settings(self):
		""" basic settings """
		
		self.namespaces = ['par', 'sol','sim'] # must be numba-able
		#self.other_attrs = [] 

	def setup(self):
		""" choose parameters """
		par = self.par
		sol = self.sol
		sim = self.sim

		# data = self.data
		# # Data
		# # get the data
		# data.data = loadmat('Data/Moments_hazard.mat')

		# # Access the 'Moments' table
		# data.moments = data.data['Moments']
		# # Determine the number of elements in moments_table
		# data.num_elements = data.moments.shape[0]

		# # Calculate the number of elements to include in moments_before
		# data.num_elements_before = data.num_elements // 2
		# # Create moments_before containing exactly half the elements in moments_table
		# data.moments_before = data.moments[1:data.num_elements_before]
		# data.moments_after = data.moments[data.num_elements_before+1:]

		# # Access the 'VCcontrols' table
		# data.vc_controls = data.data['VCcontrol']
		# data.vc_controls_before = data.vc_controls[1:data.num_elements_before, 1:data.num_elements_before]
		# data.vc_controls_after = data.vc_controls[data.num_elements_before+1:, data.num_elements_before+1:]
				


		# model
		par.full_sample_estimation = False

		# a. model
		par.euler = False  # Euler equation or optimizer
  		
		par.N = 10 #Number of reference periods
		par.M = 10 #Number of ekstra periods to reach stationary state
		# Transfers Structure
		par.T1 = 10   #Time with high transfers
		par.T2 = 10   #Time with medium transfers
		par.T3 = par.N+1 #Time with low transfers
		par.T = par.T1 + par.T2 + par.T3 + par.M #Total number of periods
		par.T_sim = 35 #Number of periods in the simulation

		par.Na = 20  #Number of grid points for savings
		
        # Income Structure
		par.w = 1.0     #Normalize wages
		par.b1 = 0.7*par.w    # High transfers
		par.b2 = 0.5*par.w    # Medium transfers
		par.b3 = 0.4*par.w    # Low transfers

		# Preferences
		par.eta = 1.0  ### Reference dependence parameter
		par.sigma = 1.0  ### Lambda in the paper
		par.delta = 0.995  ### Discount factor

		#Savings
		par.R = 1/par.delta    #Interest rate
		par.A_0 = 0.0  #Initial assets 
		par.L = -2.0  # borrowing constraint


		# EGM does not give the same as VFI if R is not 1/delta, if eta is different from zero, or if sigma is different from 1
		# It gives the same if eta = 0 or eta = anything and sigma = 1 as long as R = 1/delta
		
		
		
	

		par.Nstates_fixed = 0 # number of fixed states
		par.Nstates_fixed_pd = 0 # number of fixed post-decision states
		par.Nstates_dynamic = 2 # number of dynamic states (Employed/Unemployed)
		par.Nstates_dynamic_pd = 2 # number of dynamic post-decision states (Employed/Unemployed)
		par.Nactions = 1 # number of actions (Search effort)




		par.cost1 = 107.0
		par.cost2 = 310.4
		par.cost3 = 570.0
		par.gamma = 0.06


		par.types = 1

		
		par.type_shares1 = 0.17
		par.type_shares3 = 0.0
	
		

	def allocate(self):
		""" allocate arrays  """

		# a. unpack
		par = self.par
		sol = self.sol
		sim = self.sim
		
		par.a_grid = np.linspace(par.L, par.A_0, par.Na)  #Grid for savings

		par.type_shares2 = 1-par.type_shares1 - par.type_shares3

        #Income when unemployed
		par.income_u = np.zeros(par.T) 
		par.income_u[0:par.T1] = par.b1
		par.income_u[par.T1:par.T1+par.T2] = par.b2
		par.income_u[par.T1+par.T2:] = par.b3
	
        #Income when employed
		par.income_e = np.zeros((par.T, par.T))
		for t in range(par.T):
			par.income_e[t, :] = par.income_u
			par.income_e[t, t:] = par.w
	
	
		# Reference points unemployed
		par.r_u = np.zeros(par.T)	
		par.ref_income_u = np.zeros(par.T+par.N)
		par.ref_income_u[0:par.N] = par.w
		par.ref_income_u[par.N:] = par.income_u
		
		for t in range(par.T):
			par.r_u[t] = par.ref_income_u[t:t+par.N].mean()
	
		# Reference points employed
		par.r_e = np.zeros((par.T, par.T+par.N))
		par.ref_income_e = np.zeros((par.T, par.T+2*par.N))
		for t in range(par.T):
			par.ref_income_e[t, 0:par.N] = par.w
			par.ref_income_e[t, par.N:par.T+par.N] = par.income_e[t, :]
			par.ref_income_e[t, par.T+par.N:] = par.w
			for s in range(par.T+par.N):
				par.r_e[t, s] = par.ref_income_e[t, s:s+par.N].mean()
		
		#reference point for next ten periods
		par.r_e_future = np.zeros((par.T, par.N))
		for t in range(par.T):
			par.r_e_future[t, :] = par.r_e[t, t:t+par.N]
		
		#Reference point + m periods
		tuple = (par.T, par.T+par.N+par.M)
		par.r_e_m = np.zeros(tuple)
		for t in range(par.T):
			par.r_e_m[t, :par.T+par.N] = par.r_e[t, :par.T+par.N]
			par.r_e_m[t, par.T+par.N:] = par.w


		# Container for value functions
		par.V_e_t_a = np.zeros((par.types, par.T, par.Na))
		par.V_e = np.zeros((par.types, par.T, par.N+par.M, par.Na))

		sol.s = np.zeros((par.types, par.T, par.Na))  # Policy function search effort
		sol.a_next = np.zeros((par.types, par.T, par.Na))  # Policy function savings
		sol.c = np.zeros((par.types, par.T, par.Na))

		sol.a_next_e = np.zeros((par.types, par.T, par.N+par.M, par.Na))  # Policy function savings employed
		sol.c_e = np.zeros((par.types, par.T, par.N+par.M, par.Na))

		sim.s = np.zeros((par.types, par.T))  # Search effort
		sim.c = np.zeros((par.types, par.T)) # Consumption
		sim.a = np.zeros((par.types, par.T))  # Savings
		sim.a_e = np.zeros((par.types, par.T,par.N+par.M))
		sim.c_e = np.zeros((par.types, par.T,par.N+par.M))

		sim.s_total = np.zeros(par.T)  # Total search effort

		# b. states
		par.Nstates = par.Nstates_dynamic + par.Nstates_fixed # number of states
		par.Nstates_pd = par.Nstates_dynamic_pd + par.Nstates_fixed_pd # number of post-decision states

		par.Nstates_t = par.T # number of auxiliary states
		par.Nstates_pd_t = par.T # number of auxiliary post-decision states


		
		

	def solve(self):
		value_function_employment(self.par, self.sol)
		solve_search_and_consumption(self.par, self.sol)
		sim_search_effort(self.par, self.sol, self.sim)