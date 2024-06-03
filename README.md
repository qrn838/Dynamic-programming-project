# Dynamic Behavioral Model of Job Search 
*Dynamic programming project*

This project follows the paper "Reference-Dependent Job Search: Evidence From Hungary" by DellaVigna, Lindner, Reizer, and Schmieder (2017).

## Included files
- 1 HTM and ConSav: This notebook compares simulations of the hand-to-mouth and consumption-saving model for a given set of model parameters.
- 1 VFI and EGM: This notebook compares simulations of the consumption-savings model using either value function iteration or EGM.
- 2 Estiation - ConSav: This notebook estimates the consumption-savings model and plots relevant figures.
- 2 Estimation - HtM: This notebook estimates the HtM model and plots relevant figures.
- estimation.py: This file contains the script used for estimating the models.
- Funcs.py: This file contains relevant functions, i.e. the utility function, cost function, and their derivatives.
- Model.py: This file sets up the model class. NB! Uses EconModel class.
- solve_consumption_savings.py: This file contains the functions used to solve the consumption-savings model.
- solve_hand_to_mouth.py: This file contains the functions used to solve the HtM model. 

## Required packages

Running the model requires standard python packages.
Further requirements to run model:
- Besides standard python packages one needs to install the EconModel class. This can be done by running %pip install EconModel from the terminal
