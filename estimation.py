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


def updatepar(par, parnames, parvals):
    ''' Update parameter values in par of parameters in parnames '''

    for i,parval in enumerate(parvals):
        parname = parnames[i]
        setattr(par,parname,parval) # It gives the attibute parname the new value parval, within the par class
        if parname == 'N':
            '''If variable name is N set to integer'''
            setattr(par,parname,int(parval))
    return par



def method_simulated_moments(model,est_par,theta0, bounds):
    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'

    if model.par.full_sample_estimation == True:
            # Estimate
        obj_fun = lambda x: sum_squared_diff_moments_before_and_after(x,model,est_par)
        res = minimize(obj_fun,theta0, method='SLSQP', bounds = bounds)
    else:
        # Estimate
        obj_fun = lambda x: sum_squared_diff_moments(x,model,est_par)
        res = minimize(obj_fun,theta0, method='SLSQP', bounds = bounds)

    return res


def sum_squared_diff_moments(theta,model,est_par):

    #Update parameters
    par = model.par
    data = model.data
    par = updatepar(par,est_par,theta)


    # Solve the model before
    model.allocate()
    moments =  model.solve() 
    
    # Objective function
    weight_mat = data.vc_controls_before   
    
    moments_after = data.moments_before
    # print(np.shape(moments_after))
    moments_after = moments_after.reshape(35)

    diff = (moments-moments_after)
   
    res = (diff.T @ np.eye(35) @ diff)*100
     
    return res




def sum_squared_diff_moments_before_and_after(theta,model,est_par):

    #Update parameters
    par = model.par
    data = model.data
    par = updatepar(par,est_par,theta)

    # Solve the model before
    par.b1 = 222/675*par.w
    par.b2 = par.b1
    model.allocate()
    moments_before_model =  model.solve() 
    # print(np.shape(moments_before_model))

    # Solve model after
    par.b1 = 342.0/675.0
    par.b2 = 171.0/675.0
    model.allocate()
    moments_after_model = model.solve()
    # print(np.shape(moments_after_model))

    model_moments = np.concatenate((moments_before_model, moments_after_model))
    # print(np.shape(model_moments))

    # rows_before, cols_before = data.vc_controls_before.shape
    # rows_after, cols_after = data.vc_controls_after.shape

    # weight_mat = np.zeros((rows_before + rows_after, cols_before + cols_after))

    # weight_mat[:rows_before, :cols_before] =  data.vc_controls_before
    # weight_mat[rows_after:, cols_after:] = data.vc_controls_after  

    
    moments_before = data.moments_before
    moments_before = moments_before.reshape(35)
    # print(np.shape(moments_before))

    moments_after = data.moments_after
    moments_after = moments_after.reshape(35)
    # print(np.shape(moments_after))

    data_moments = np.concatenate((moments_before, moments_after))
    # print(np.shape(data_moments))


    diff = (model_moments-data_moments)
   
    res = (diff.T @ np.eye(70) @ diff)*100

    #Set paramters to before reform again
    par.b1 = 222/675*par.w
    par.b2 = par.b1
    model.allocate()
     
    return res

    
    

		