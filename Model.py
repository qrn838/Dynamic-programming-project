from types import SimpleNamespace
import numpy as np
import scipy
from copy import deepcopy

from EconModel import EconModelClass


class ReferenceDependenceClass(EconModelClass):

	def settings(self):
		""" basic settings """
		
		self.namespaces = ['par'] # must be numba-able   R: Hvad fanden betyder det her?
		#self.other_attrs = [] 

	def setup(self):
		""" choose parameters """
		par = self.par
		# a. model
  		
		par.N = 15 #Number of reference periods
		# Transfers Structure
		par.T1 = 10   #Time with high transfers
		par.T2 = 10   #Time with medium transfers   R: Saa altsaa foer front loading eller hvad?
		par.T3 = par.N+1 #Time with low transfers
		par.T = par.T1 + par.T2 + par.T3 #Total number of periods
		
        # Income Structure
		par.w = 1.0     #Normalize wages
		par.b1 = 0.7*par.w    # High transfers
		par.b2 = 0.5*par.w    # Medium transfers
		par.b3 = 0.4*par.w    # Low transfers

		# Preferences
		par.eta = 1.0	 # Captures reference point
		par.sigma = 1.5  # Lambda in the paper, i.e. loss aversion
		par.delta = 0.9  # Discount factor
	

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
		par.income_u = np.zeros(par.T)					# Empty array to store benefits
		par.income_u[0:par.T1] = par.b1					# Benefits in first T1 periods (high benefits)
		par.income_u[par.T1:par.T1+par.T2] = par.b2		# Benefite in middle T2 periods (medium benefits)
		par.income_u[par.T1+par.T2:] = par.b3			# Benefits in last T3 periods (low benefits)
	
        #Income when employed
		par.income_e = np.zeros((par.T, par.T))			# Empty array to store income
		for t in range(par.T):
			par.income_e[t, :] = par.income_u			# Income if unemployed
			par.income_e[t, t:] = par.w					# Income after finding job
	
	
		# Reference points unemployed
		par.r_u = np.zeros(par.T)						# Reference point given by last N periods income (page 1980 in DellaVigna)
		# R: Jeg er ikke helt sikker på, hvad de næste 3 er?
		par.ref_income_u = np.zeros(par.T+par.N)		# Stores the income history of unemployed individuals. 
		par.ref_income_u[0:par.N] = par.w				# Some buffer zone?
		par.ref_income_u[par.N:] = par.income_u			# Stores actual income levels for unemployed individuals
		
		for t in range(par.T):
			par.r_u[t] = par.ref_income_u[t:t+par.N].mean()		# Calculates the reference point for unemployed individuals by taking the mean of the income over the last N periods. 
	
		# Reference points employed
		# Notice that we now use two dimensions. This is because, we need to account for both wage when employed and the benefit level before finding a job 
		par.r_e = np.zeros((par.T, par.T+par.N))				# reference point given by last N periods income (page 1980 in DellaVigna)
		par.ref_income_e = np.zeros((par.T, par.T+2*par.N))		# Stores the income history of employed individuals R: Hvorfor 2 gange N?
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
	

		# b. states
		par.Nstates = par.Nstates_dynamic + par.Nstates_fixed			 # number of states
		par.Nstates_pd = par.Nstates_dynamic_pd + par.Nstates_fixed_pd 	 # number of post-decision states

		par.Nstates_t = par.T 		# number of auxiliary states
		par.Nstates_pd_t = par.T 	# number of auxiliary post-decision states
		

