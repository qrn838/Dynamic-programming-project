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



def method_simulated_moments(model,est_par,theta0, bounds,weight=True):
    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'

    # Estimate
    obj_fun = lambda x: sum_squared_diff_moments(x,model,est_par,weight)
    res = minimize(obj_fun,theta0, method='nelder-mead', bounds = bounds)

    return res

# Estimerer på både før og efter
# def sum_squared_diff_moments(theta,model,est_par,weight=True):

#     #Update parameters
#     par = model.par
#     data = model.data
#     par = updatepar(par,est_par,theta)

#     # Solve the model
#     model.allocate()
    
#     # Objective function
#     weight_mat_before = data.vc_controls_before
#     weight_mat_after = data.vc_controls_after 

#     moments_cal =  model.solve()   

#     moments_before = data.moments_before
#     moments_before = moments_before.reshape(35)

#     moments_after = data.moments_after
#     moments_after = moments_after.reshape(35)

#     diff1 = (moments_cal-moments_before)

#     if weight==True:
#         res1 = (diff1.T @ weight_mat_before @ diff1)
#     else:
#         res1 = (diff1.T @ diff1)

#     diff2 = (moments_cal-moments_after)

#     if weight==True:
#         res2 = (diff2.T @ weight_mat_after @ diff2)
#     else:
#         res2 = (diff2.T @ diff2)

#     tot_res = res1 + res2

#     # # Calculate squared differences
#     # res_set1 = np.sum(diff1 ** 2)
#     # res_set2 = np.sum(diff2 ** 2)

#     # # Return total sum of squared differences for both sets
#     # total_res = res_set1 + res_set2
     
#     return tot_res

#Estimerer kun på før / efter
def sum_squared_diff_moments(theta,model,est_par,weight):

    #Update parameters
    par = model.par
    data = model.data
    par = updatepar(par,est_par,theta)

    # Solve the model
    model.allocate()
    
    # Objective function
    weight_mat = data.vc_controls_before  
    moments =  model.solve()   

    moments_after = data.moments_before
    # print(np.shape(moments_after))
    moments_after = moments_after.reshape(35)

    diff = (moments-moments_after)
<<<<<<< HEAD
   
    res = (diff.T @ np.eye(35) @ diff)*100
=======

    if weight==True:
        res = (diff.T @ weight_mat @ diff) 
    else:
        res = (diff.T @ np.eye(35) @ diff)
>>>>>>> 95851a81df8b26b6dd88eab38c21c4a13fc21371
     
    return res

    
    

		