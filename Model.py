from types import SimpleNamespace
import numpy as np
import scipy
from copy import deepcopy
from scipy.io import loadmat

from EconModel import EconModelClass
from solve_hand_to_mouth import *
from solve_consumption_saving import *



class ReferenceDependenceClass(EconModelClass):

	def settings(self):
		""" basic settings """
		
		self.namespaces = ['par', 'sol','sim', 'data'] # must be numba-able


	def setup(self):
		""" choose parameters """
		par = self.par
		sol = self.sol
		sim = self.sim
		data = self.data

		####################################################
		# Data
		# get the data
		data.data = loadmat('Data/Moments_hazard.mat')

		# # Access the 'Moments' table
		data.moments = data.data['Moments']
		# # Determine the number of elements in moments_table
		data.num_elements = data.moments.shape[0]

		# # Calculate the number of elements to include in moments_before
		data.num_elements_before = data.num_elements // 2
		# # Create moments_before containing exactly half the elements in moments_table
		data.moments_before = data.moments[1:data.num_elements_before]
		data.moments_after = data.moments[data.num_elements_before+1:]

		# # Access the 'VCcontrols' table
		data.vc_controls = data.data['VCcontrol']
		data.vc_controls_before = data.vc_controls[1:data.num_elements_before, 1:data.num_elements_before]
		data.vc_controls_after = data.vc_controls[data.num_elements_before+1:, data.num_elements_before+1:]

		# Weight matrix
		data.rows_before, data.cols_before = data.vc_controls_before.shape 
		data.rows_after, data.cols_after = data.vc_controls_after.shape
		
		data.weight_mat = np.zeros((data.rows_before + data.rows_after, data.cols_before + data.cols_after))
		data.weight_mat[:data.rows_before, :data.cols_before] =  data.vc_controls_before
		data.weight_mat[data.rows_after:, data.cols_after:] = data.vc_controls_after  
		####################################################

		####### Options ########
		# Estimate full sample or only before
		par.full_sample_estimation = True

		# Use EGM to get value of getting employed
		par.euler = True  

		# Hand-to-mouth or Consumption-Saving
		par.model = 'Mixed'  # 'ConSav' or 'HTM' or 'Mixed'
		#####################
  		
		# Time Structure
		par.N = 13 #Number of reference periods
		par.M = 10 #Number of ekstra periods to reach stationary state
		# Transfers Structure
		par.T1 = 6   #Time with high transfers
		par.T2 = 12   #Time with medium transfers
		par.T3 = 6 #Time with low transfers
		par.T = par.T1 + par.T2 + par.T3 + par.N + par.M #Total number of periods
		par.T_sim = 35 #Number of periods in the simulation

		
		
        # Income Structure
		par.w = 1.0     #Normalize wages
		par.welfare = 90/675
		par.b1 = 222/675*par.w    # High transfers
		par.b2 = par.b1    # Medium transfers
		par.b3 = 114/675*par.w    # Low transfers
		par.b4 = par.welfare*par.w	# Welfare
		# par.b4 = par.b3

		# Preferences
		par.eta = 1.0  		# Reference dependence parameter
		par.lambdaa = 2.0  # Loss aversion
		par.delta = 0.995  	# Discount factor

		#Savings
		par.R = 1/par.delta + 0.001   	#Interest rate
		par.A_0 = 0.0  					#Initial assets 
		par.L = -0.5				#borrowing constraint
		par.Na = 10			#Number of grid points for savings
		par.HTM_share = 0.5  # Share of Hand-to-mouth agents in mixed model
	

		# Search costs for different types
		par.cost1 = 107.0
		par.cost2 = 310
		par.cost3 = 570.0
		par.gamma = 0.06	# Inverse of elasticity of search effort w.r.t. value of employment

		par.types = 3	# Number of types

		#Share of types (Type 2 is residual)
		par.type_shares1 = 1.0	
		par.type_shares3 = 0.0

		############################################
		####		   Initial guesses 		    ####
		############################################


		par.noSearchInits = 30			# Number of numerical minimizations to run

		# The range from initial values are drawn (The same as in DellaVigna et al. (2017))
		# Search cost 1 type
		par.lb_c = 200
		par.ub_c = 400
		# Highest search cost
		par.lb_hsc = 300
		par.ub_hsc = 700
		# Medium search cost
		par.lb_msc = 50
		par.ub_msc = 100
		# Lowest search cost
		par.lb_lsc = 0
		par.ub_lsc = 100
		# Gamma
		par.lb_gamma = 0.1
		par.ub_gamma = 1.3
		# Shares of types
		par.lb_share = 0.1
		par.ub_share = 2/par.types
		# Lambdaa
		par.lb_lambdaa = 1
		par.ub_lambdaa = 5
		# N
		par.lb_N = 10
		par.ub_N = 20
		# Welfare
		par.lb_welfare = 0
		par.ub_welfare = 200/450
		# Delta
		par.lb_delta = 0.7
		par.ub_delta = 0.999
		# L
		par.lb_L = -0.9
		par.ub_L = -0.001

		################################################
		# Needed for EconModelClass (not used)
		par.Nstates_fixed = 0 # number of fixed states
		par.Nstates_fixed_pd = 0 # number of fixed post-decision states
		par.Nstates_dynamic = 2 # number of dynamic states (Employed/Unemployed)
		par.Nstates_dynamic_pd = 2 # number of dynamic post-decision states (Employed/Unemployed)
		par.Nactions = 1 # number of actions (Search effort)
		###############################################
	
		

	def allocate(self):
		""" allocate arrays  """

		# a. Unpack variables
		par = self.par
		sol = self.sol
		sim = self.sim
		
		
		par.a_grid = np.linspace(par.L, par.A_0, par.Na)  #Grid for savings
		par.m_grid = np.linspace(par.L+par.w, par.A_0+par.w, par.Na)  #Grid for cash-on-hand

		par.type_shares2 = 1-par.type_shares1 - par.type_shares3

	

        #Income when unemployed
		par.income_u = np.zeros(par.T)					# Empty array to store benefits
		par.income_u[0:par.T1] = par.b1					# Benefits in first T1 periods (high benefits)
		par.income_u[par.T1:par.T1+par.T2] = par.b2		# Benefite in middle T2 periods (medium benefits)
		par.income_u[par.T1+par.T2:par.T1+par.T2+ par.T3] = par.b3			# Benefits in last T3 periods (low benefits)
		par.income_u[par.T1+par.T2+par.T3:] = par.b4						# Benefits after T3 periods (welfare)


		# Reference points when unemployed taking wages before unemployment into account
		par.r_u = np.zeros(par.T)							# Actual reference point, i.e. mean of income history
		par.ref_income_u = np.zeros(par.T+int(par.N))		# Income path when unemployed
		par.ref_income_u[0:int(par.N)] = par.w
		par.ref_income_u[int(par.N):] = par.income_u
		for t in range(par.T):
			par.r_u[t] = par.ref_income_u[t:t+int(par.N)].mean()


		# Income path when getting employed at time t
		par.income_e = np.zeros((par.T, par.T))			
		for t in range(par.T):
			par.income_e[t, :] = par.income_u
			par.income_e[t, t:] = par.w
	
		# Reference points when employed taking time of employment into account
		par.r_e = np.zeros((par.T, par.T+int(par.N)))		 		# Actual reference point, i.e. mean of income history
		par.ref_income_e = np.zeros((par.T, par.T+2*int(par.N)))	# Income path when employed
		for t in range(par.T):
			par.ref_income_e[t, 0:int(par.N)] = par.w
			par.ref_income_e[t, int(par.N):par.T+int(par.N)] = par.income_e[t, :]
			par.ref_income_e[t, par.T+int(par.N):] = par.w
			for s in range(par.T+int(par.N)):
				par.r_e[t, s] = par.ref_income_e[t, s:s+int(par.N)].mean()
		
		# Reference point for next N periods used in HtM for calculating value of getting employed
		par.r_e_future = np.zeros((par.T, int(par.N)))
		for t in range(par.T):
			par.r_e_future[t, :] = par.r_e[t, t:t+int(par.N)]
		
		# Reference point after employment + m periods to ensure stationary state (last N periods r = w)
		tuple = (par.T, par.T+int(par.N)+par.M)
		par.r_e_m = np.zeros(tuple)
		for t in range(par.T):
			par.r_e_m[t, :par.T+int(par.N)] = par.r_e[t, :]
			par.r_e_m[t, par.T+int(par.N):] = par.w


		# Container for value functions
		par.V_e_t_a = np.zeros((par.T, par.Na))				# Value of getting employed at time t
		par.V_e = np.zeros((par.T, par.N+par.M, par.Na))		# Value function when employed

		sol.s = np.zeros((par.types, par.T, par.Na))  		# Policy function search effort
		sol.a_next = np.zeros((par.types, par.T, par.Na))  	# Policy function savings
		sol.c = np.zeros((par.types, par.T, par.Na))		# Policy function consumption

		sol.a_next_e = np.zeros((par.T, par.N+par.M, par.Na))  # Policy function savings employed
		sol.c_e = np.zeros((par.T, par.N+par.M, par.Na))		  # Policy function consumption employed

		sim.s = np.zeros((par.types, par.T))  				# Search effort for each type
		sim.s_total = np.zeros(par.T_sim)  					# Total search effort
		sim.a_next = np.zeros((par.types, par.T))  			# Savings for each type
		sim.c = np.zeros((par.types, par.T)) 				# Consumption for each type
		sim.a = np.zeros((par.types, par.T))  				# Current assets for each type
		sim.a_e = np.zeros((par.T,par.N+par.M))	# Current assets for each type when employed
		sim.c_e = np.zeros((par.T,par.N+par.M))	# Consumption for each type when employed



		################################################
		# Needed for EconModelClass (not used)
		par.Nstates = par.Nstates_dynamic + par.Nstates_fixed # number of states
		par.Nstates_pd = par.Nstates_dynamic_pd + par.Nstates_fixed_pd # number of post-decision states

		par.Nstates_t = par.T # number of auxiliary states
		par.Nstates_pd_t = par.T # number of auxiliary post-decision states
		###############################################
		
		

	def solve_ConSav(self):
		""" solve the model for Consumption-Saving agent"""
		value_function_employment_ConSav(self.par, self.sol)
		solve_search_and_consumption_ConSav(self.par, self.sol)
		sim_search_effort_ConSav(self.par, self.sol, self.sim)
		sim_s = self.sim.s_total
		return sim_s
	
	def solve_HTM(self):
		""" solve the model for Hand-to-Mouth agent"""
		sim_s = sim_search_effort_HTM(self.par)
		return sim_s
	
	def solve(self):
		""" solve the model """
		if self.par.model == 'ConSav':
			sim_s = self.solve_ConSav()
		elif self.par.model == 'HTM':
			sim_s = self.solve_HTM()
		else:
			sim_s_consav = self.solve_ConSav()
			sim_s_htm = self.solve_HTM()
			sim_s = self.par.HTM_share*sim_s_htm + (1-self.par.HTM_share)*sim_s_consav
		return sim_s

