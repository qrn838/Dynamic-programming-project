from types import SimpleNamespace
import numpy as np
import scipy
import numpy as np
import copy
from scipy.optimize import minimize
import scipy.optimize as optimize

from Funcs import *
from solve_hand_to_mouth import *
from solve_consumption_saving import *
from scipy.io import loadmat


def updatepar(par, parnames, parvals):
    ''' Update parameter values in par of parameters in parnames '''

    for i,parval in enumerate(parvals):
        parname = parnames[i]
        setattr(par,parname,parval)     # It gives the attibute parname the new value parval, within the par class
        if parname == 'N':
            '''If variable name is N set to integer'''
            setattr(par,parname,int(parval))
    return par



def method_simulated_moments(model,est_par,theta0, bounds, weight):
    ''' Estimate the model using simulated moments'''
    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'

    if model.par.full_sample_estimation == True:  
        # Estimate using the full sample
        obj_fun = lambda x: sum_squared_diff_moments_before_and_after(x,model,est_par,weight)
        res = minimize(obj_fun,theta0, method='SLSQP', bounds = bounds)
    else:
        # Estimate using only the before reform
        obj_fun = lambda x: sum_squared_diff_moments(x,model,est_par,weight)
        res = minimize(obj_fun,theta0, method='SLSQP', bounds = bounds)

    return res


def sum_squared_diff_moments(theta,model,est_par,weight=False):
    ''' Objective function for estimating the model before the reform using simulated moments'''
    #Update parameters
    par = model.par
    data = model.data
    par = updatepar(par,est_par,theta)


    # Solve the model
    model.allocate()
    if par.model == 'HTM':
        moments = model.solve_HTM()
    elif par.model == 'ConSav':
        moments = model.solve_ConSav()
    else:
        moments = model.solve()
    
    # Objective function
    weight_mat = data.vc_controls_before   
    moments_after = data.moments_before
    moments_after = moments_after.reshape(36)

    diff = (moments-moments_after)
   
    if weight:    # Weights are used
        res = (diff.T @ weight_mat @ diff)*100 #res = (diff.T @ np.linalg.inv(weight_mat) @ diff)*100
    else:         # Identity matrix is used
        res = (diff.T @ np.eye(35) @ diff)*100
     
    return res



def model_moments_combined(model,est_par,theta):
    

    #Update parameters
    par = model.par
    data = model.data
    par = updatepar(par,est_par,theta)

    # Solve the model before
    par.b1 = 222/675*par.w
    par.b2 = par.b1
    model.allocate()
    if par.model == 'HTM':
        moments_before_model =  model.solve_HTM()
    elif par.model == 'ConSav':
        moments_before_model = model.solve_ConSav() 
    else:
        moments_before_model = model.solve()

    # Solve model after
    par.b1 = 342.0/675.0
    par.b2 = 171.0/675.0
    model.allocate()
    if par.model == 'HTM':
        moments_after_model = model.solve_HTM()
    elif par.model == 'ConSav':
        moments_after_model = model.solve_ConSav()
    else:
        moments_after_model = model.solve()

    # Combine model results from before and after refor
    model_moments = np.concatenate((moments_before_model, moments_after_model)) 

    #Set paramters to before reform again
    par.b1 = 222/675*par.w
    par.b2 = par.b1
    model.allocate() 

    return model_moments


def sum_squared_diff_moments_before_and_after(theta,model,est_par,weight=True):
    ''' Objective function for estimating the model on the full sample using simulated moments'''

    #Update parameters
    par = model.par
    data = model.data

    model_moments = model_moments_combined(model,est_par,theta)   


    ###### Combine data moments from before and after reform ###########

    weight_mat = data.weight_mat
    
    moments_before = data.moments_before
    moments_before = moments_before.reshape(35)

    moments_after = data.moments_after
    moments_after = moments_after.reshape(35)

    data_moments = np.concatenate((moments_before, moments_after))
    ###############################################################


    diff = (model_moments-data_moments)

    if weight:      # Weights are used
        res = (diff.T @ weight_mat @ diff)*100 #res = (diff.T @ np.linalg.inv(weight_mat) @ diff)*100
    else:           # Identity matrix is used
        res = (diff.T @ np.eye(70) @ diff)*100

    return res

def getSearchInits_benchmark_HTM(model):
    par = model.par

    lb = np.array(par.lb_rep.T,par.noSearchInits)
    ub = np.arry(par.ub_rep.T,par.noSearchInits)
    searchInits = lb + (ub - lb) * np.random(par.noOfParams,par.noSearchInits)

    return searchInits

    
    

		