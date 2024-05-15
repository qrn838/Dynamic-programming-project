from types import SimpleNamespace
import numpy as np
import scipy
from copy import deepcopy
from scipy.io import loadmat

from hand_to_mouth import *

from EconModel import EconModelClass


class ReferenceDependenceClass(EconModelClass):

	def settings(self):
		""" basic settings """
		
		self.namespaces = ['par', 'data'] 
		#self.other_attrs = [] 

	def setup(self):
		""" choose parameters """
		par = self.par
		data = self.data
		# Data
		# get the data
		data.data = loadmat('Data/Moments_hazard.mat')

		# Access the 'Moments' table
		data.moments = data.data['Moments']
		# Determine the number of elements in moments_table
		data.num_elements = data.moments.shape[0]

		# Calculate the number of elements to include in moments_before
		data.num_elements_before = data.num_elements // 2
		# Create moments_before containing exactly half the elements in moments_table
		data.moments_before = data.moments[1:data.num_elements_before]
		data.moments_after = data.moments[data.num_elements_before+1:]

		# Access the 'VCcontrols' table
		data.vc_controls = data.data['VCcontrol']
		data.vc_controls_before = data.vc_controls[1:data.num_elements_before, 1:data.num_elements_before]
		data.vc_controls_after = data.vc_controls[data.num_elements_before+1:, data.num_elements_before+1:]
				


		# model

		par.N = 15 #Number of reference periods
		par.M = 10 #Number of periods in the future to ensure convergence
		# Transfers Structure
		par.T1 = 6   #Time with high transfers
		par.T2 = 12   #Time with medium transfers   R: Saa altsaa foer front loading eller hvad? S: Det gør det bare muligt at lave både front loading
		par.T3 = par.N+par.M #Time with low transfers
		par.T =  par.T1 + par.T2 + par.T3  #Total number of periods
		par.T_sim = 35 #Number of periods in the simulation
		
        # Income Structure
		par.w = 1.0     		    #Normalize wages
		# par.b1 = 342/675*par.w    # High transfers
		# par.b2 = 171/675*par.w    # Medium transfers
		par.b3 = 114/675*par.w      # Low transfers
		par.b1 = 223/675*par.w    
		par.b2 = par.b1

		# Preferences
		par.eta = 1.0	   # Captures reference dependence
		par.sigma = 2.23   # Lambda in the paper, i.e. loss aversion
		par.delta = 0.995  # Discount factor
	

		par.Nstates_fixed = 0 # number of fixed states
		par.Nstates_fixed_pd = 0 # number of fixed post-decision states
		par.Nstates_dynamic = 2 # number of dynamic states (Employed/Unemployed)
		par.Nstates_dynamic_pd = 2 # number of dynamic post-decision states (Employed/Unemployed)
		par.Nactions = 1 # number of actions (Search effort)
	
		
		par.cost1 = 107.0
		par.cost2 = 310.4
		par.gamma = 0.06
		par.types = 2

		
		par.type_shares1 = 0.17
		

		
		

	def allocate(self):
		""" allocate arrays  """

		# a. unpack
		par = self.par

		par.type_shares2 = 1-par.type_shares1
		
        #Income when unemployed
		par.income_u = np.zeros(par.T)				# Empty array to store benefits
		par.income_u[0:par.T1] = par.b1					# Benefits in first T1 periods (high benefits)
		par.income_u[par.T1:par.T1+par.T2] = par.b2		# Benefite in middle T2 periods (medium benefits)
		par.income_u[par.T1+par.T2:] = par.b3			# Benefits in last T3 periods (low benefits)
	
        #Income when employed
		par.income_e = np.zeros((par.T, par.T))		# Empty array to store income
		for t in range(par.T):
			par.income_e[t, :] = par.income_u			# Income if unemployed
			par.income_e[t, t:] = par.w				# Income after finding job SHOULD IT BE t+1: ?
	
	
		# Reference points unemployed
		par.r_u = np.zeros(par.T)						# Reference point given by last N periods income (page 1980 in DellaVigna)

		par.ref_income_u = np.zeros(par.T+int(par.N))		# Stores the income history of unemployed individuals. 
		par.ref_income_u[0:int(par.N)] = par.w				# Some buffer zone? S: I første periode af arbejdsløshed er referencepointet givet ved lønnen
		par.ref_income_u[int(par.N):] = par.income_u			# Stores actual income levels for unemployed individuals
		
		for t in range(par.T):
			par.r_u[t] = par.ref_income_u[t:t+int(par.N)].mean()		# Calculates the reference point for unemployed individuals by taking the mean of the income over the last N periods. 
	
		# Reference points employed
		# Notice that we now use two dimensions. This is because, we need to account for both wage when employed and the benefit level before finding a job 
		par.r_e = np.zeros((par.T, par.T+int(par.N)))				# reference point given by last N periods income (page 1980 in DellaVigna)
		par.ref_income_e = np.zeros((par.T, par.T+2*int(par.N)))		# Stores the income history of employed individuals R: Hvorfor 2 gange N?
		for t in range(par.T):
			par.ref_income_e[t, 0:int(par.N)] = par.w
			par.ref_income_e[t, int(par.N):par.T+int(par.N)] = par.income_e[t, :]
			par.ref_income_e[t, par.T+int(par.N):] = par.w
			for s in range(par.T+int(par.N)):
				par.r_e[t, s] = par.ref_income_e[t, s:s+int(par.N)].mean()
		
		#reference point for next ten periods
		par.r_e_future = np.zeros((par.T, int(par.N)))
		for t in range(par.T):
			par.r_e_future[t, :] = par.r_e[t, t:t+int(par.N)]
	

		# b. states
		par.Nstates = par.Nstates_dynamic + par.Nstates_fixed			 # number of states
		par.Nstates_pd = par.Nstates_dynamic_pd + par.Nstates_fixed_pd 	 # number of post-decision states

		par.Nstates_t = par.T 		# number of auxiliary states
		par.Nstates_pd_t = par.T 	# number of auxiliary post-decision states
		

	def solve(self):
		sim_s = sim_search_effort(self.par)
		return sim_s


