from types import SimpleNamespace
import numpy as np
import scipy
import numpy as np
import copy
from scipy.optimize import minimize
import scipy.optimize as optimize

from Funcs import *
from hand_to_mouth import *
from scipy.io import loadmat
import Model


def updatepar(par, parnames, parvals):
    ''' Update parameter values in par of parameters in parnames '''

    for i,parval in enumerate(parvals):
        parname = parnames[i]
        setattr(par,parname,parval) # It gives the attibute parname the new value parval, within the par class
    return par



def method_simulated_moments(model,est_par,theta0, bounds):
    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'

    # Estimate
    obj_fun = lambda x: sum_squared_diff_moments(x,model,est_par)
    res = minimize(obj_fun,theta0, method='nelder-mead', bounds = bounds)

    return res


def sum_squared_diff_moments(theta,model,est_par):

    #Update parameters
    par = model.par
    data = model.data
    par = updatepar(par,est_par,theta)

    # Solve the model
    model.allocate()
    
    # Objective function
    weight_mat = data.vc_controls_after   
    moments =  model.solve()   

    moments_after = data.moments_after
    moments_after = moments_after.reshape(35)

    diff = (moments-moments_after)
   
    res = (diff.T @ weight_mat @ diff)*1000
     
    return res

    
    

		