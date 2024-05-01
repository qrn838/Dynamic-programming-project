from types import SimpleNamespace
import numpy as np
import scipy
from copy import deepcopy

from EconModel import EconModelClass


class ReferenceDependenceClass(EconModelClass):

	def settings(self):
		""" basic settings """
		
		self.namespaces = ['par'] # must be numba-able
		#self.other_attrs = [] 

	def setup(self):
		""" choose parameters """
		par = self.par
		# a. model
  		
		par.N = 15 #Number of reference periods
		par.M = 20 #Number of ekstra periods to reach stationary state
		# Transfers Structure
		par.T1 = 10   #Time with high transfers
		par.T2 = 10   #Time with medium transfers
		par.T3 = par.N+1 #Time with low transfers
		par.T = par.T1 + par.T2 + par.T3 #Total number of periods
		
        # Income Structure
		par.w = 1.0     #Normalize wages
		par.b1 = 0.5*par.w    # High transfers
		par.b2 = 0.45*par.w    # Medium transfers
		par.b3 = 0.4*par.w    # Low transfers

		# Preferences
		par.eta = 0.5  ### Reference dependence parameter
		par.sigma = 1.5  ### Lambda in the paper
		par.delta = 0.9  ### Discount factor

		#Savings
		par.R = 1/par.delta-0.05  #Interest rate
		par.A_0 = 0.0  #Initial assets 
		par.L = -1.0  # borrowing constraint
		par.Na = 20  #Number of grid points for savings
		par.a_grid = np.linspace(par.L, par.A_0, par.Na)  #Grid for savings
	

		par.Nstates_fixed = 0 # number of fixed states
		par.Nstates_fixed_pd = 0 # number of fixed post-decision states
		par.Nstates_dynamic = 2 # number of dynamic states (Employed/Unemployed)
		par.Nstates_dynamic_pd = 2 # number of dynamic post-decision states (Employed/Unemployed)
		par.Nactions = 1 # number of actions (Search effort)
	
		

	def allocate(self):
		""" allocate arrays  """

		# a. unpack
		par = self.par
		
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
		par.V_e_t_a = np.zeros((par.T, par.Na))


		# b. states
		par.Nstates = par.Nstates_dynamic + par.Nstates_fixed # number of states
		par.Nstates_pd = par.Nstates_dynamic_pd + par.Nstates_fixed_pd # number of post-decision states

		par.Nstates_t = par.T # number of auxiliary states
		par.Nstates_pd_t = par.T # number of auxiliary post-decision states
		

