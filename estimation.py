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



def method_simulated_moments(model,est_par,theta0, bounds, weight):
    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'

    if model.par.full_sample_estimation == True:
            # Estimate
        obj_fun = lambda x: sum_squared_diff_moments_before_and_after(x,model,est_par,weight)
        res = minimize(obj_fun,theta0, method='SLSQP', bounds = bounds)
    else:
        # Estimate
        obj_fun = lambda x: sum_squared_diff_moments(x,model,est_par,weight)
        res = minimize(obj_fun,theta0, method='SLSQP', bounds = bounds)

    return res


def sum_squared_diff_moments(theta,model,est_par,weight):

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
    moments_after = moments_after.reshape(36)

    diff = (moments-moments_after)
   
    if weight:
        res = (diff.T @ weight_mat @ diff)*100 #res = (diff.T @ np.linalg.inv(weight_mat) @ diff)*100
    else:
        res = (diff.T @ np.eye(35) @ diff)*100
     
    return res




def sum_squared_diff_moments_before_and_after(theta,model,est_par,weight):

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

    model_moments = np.concatenate((moments_before_model, moments_after_model))

    rows_before, cols_before = data.vc_controls_before.shape
    rows_after, cols_after = data.vc_controls_after.shape

    weight_mat = np.zeros((rows_before + rows_after, cols_before + cols_after))

    weight_mat[:rows_before, :cols_before] =  data.vc_controls_before
    weight_mat[rows_after:, cols_after:] = data.vc_controls_after  

    
    moments_before = data.moments_before
    moments_before = moments_before.reshape(35)

    moments_after = data.moments_after
    moments_after = moments_after.reshape(35)

    data_moments = np.concatenate((moments_before, moments_after))


    diff = (model_moments-data_moments)

    if weight:
        res = (diff.T @ weight_mat @ diff)*100 #res = (diff.T @ np.linalg.inv(weight_mat) @ diff)*100
    else:
        res = (diff.T @ np.eye(70) @ diff)*100

    #Set paramters to before reform again
    par.b1 = 222/675*par.w
    par.b2 = par.b1
    model.allocate()
     
    return res

# Matlab kode de bruger til at trække initiat guesses

# %number of parameters
#   noOfParams=131;
# %number of numerical minimizations to run
#   noSearchInits=300;

# %The range from initial values are drawn, also shouldn't be changed
# % THESE ARE NOT SUPPOSED TO BE CHANGED 
# % VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV%
# lb_rep = zeros(1,noOfParams);
# ub_rep = ones(1,noOfParams);

# lb_rep(2)= 50;      ub_rep(2) = 1000;   %k_1 - lowest type (highest cost)
# lb_rep(3)= 30;      ub_rep(3) = 100;    %k_2        |
# lb_rep(4)= 0;       ub_rep(4) = 50;     %k_3        |
# lb_rep(5)= 0;       ub_rep(5) = 30;     %k_4        | 
# lb_rep(6)= 0;       ub_rep(6) = 10;     %k_5        V
# lb_rep(7)= 0;       ub_rep(7) = 5;      %k_6 - highest type (lowest cost)

# lb_rep(8)= 0.1;     ub_rep(8) = 1.3;    %gamma - inverse elasticity, doesn't matter if no_of_gamma_types > 1
# lb_rep(39:44)= lb_rep(8);     
# ub_rep(39:44) = ub_rep(8);    %gammas - inverse elasticity, doesn't matter if no_of_gamma_types == 1

# lb_rep(9:14)= 0;    ub_rep(9:14) = 2/no_of_types; %shares of types
# lb_rep(15)= 0;      ub_rep(15) = 3;     %kappa
# lb_rep(17)= 1;      ub_rep(17) = 30;    %lambda - loss aversion
# lb_rep(18)= 1;      ub_rep(18) = 25;    %N - periods of adjustment ..or.. 1/rho when AR==1
# lb_rep(19)= 0;      ub_rep(19) = 200;   %welfare - welfare benefits

# lb_rep(46)= 0;      ub_rep(46) = 500;   % asset grid size
# lb_rep(47)= 1;      ub_rep(47) = 50;    % asset grid jump
# lb_rep(51:53)= lb_rep(N);      ub_rep(51:53) = ub_rep(N);    % N 
# lb_rep(55:57)= lb_rep(lambda);      ub_rep(55:57) = ub_rep(lambda);    

# lb_rep_orig = lb_rep;
# ub_rep_orig = ub_rep;
# %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^%

# % Draw the initial values
# function [searchInits] = getSearchInits_benchmark(noSearchInits)
#     lb=repmat(lb_rep',1,noSearchInits);
#     ub=repmat(ub_rep',1,noSearchInits);
#     searchInits=lb+(ub-lb).*rand(noOfParams,noSearchInits);
# end

def getSearchInits_benchmark(model):
    par = model.par

    lb = np.array(par.lb_rep.T,par.noSearchInits)
    ub = np.arry(par.ub_rep.T,par.noSearchInits)
    searchInits = lb + (ub - lb) * np.random(par.noOfParams,par.noSearchInits)

    return searchInits

    
    

		