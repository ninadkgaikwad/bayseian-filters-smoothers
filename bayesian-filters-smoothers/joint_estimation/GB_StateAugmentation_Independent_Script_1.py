# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import math
import time
import tracemalloc
import copy
import scipy


## Hi Ninad
# Getting path for custom modules 
module_path = os.path.abspath(os.path.join('.','bayesian-filters-smoothers'))

# Adding custom module path to Python path
if module_path not in sys.path:
    sys.path.append(module_path)
    
# Printing the custom module paths
print('Custom Module Path:' + '\n' + module_path)

## Importing custom modules
from bayesian_filters_smoothers import Extended_Kalman_Filter_Smoother as EKFS
from bayesian_filters_smoothers import Unscented_Kalman_Filter_Smoother as UKFS
from bayesian_filters_smoothers import Gaussian_Filter_Smoother as GFS

# =============================================================================
# User Inputs
# =============================================================================
Type_Data_Source = 2  # 1 - PNNNL Prototypes , 2 - Simulated 4State House Model

Type_System_Model = 2 # 1 - One State Model , 2 - Two State Model , 4 - Four State Model

Shoter_ResultsPath = True  # True - Results are stored in shorter path , False - Results are stored in the appropriate directory

Short_ResultFolder = 'Results_1' # Name of the Folder where results will be stored

DateTime_Tuple = ('08/01/2017', '08/07/2017')

Test_Split = 0.3  # Data splitting for training and testing

FileResolution_Minutes = 10

StateAug_Filter = 1  # 1 - EKF, 2 - UKF , 3 - GF-Type1 Gauss-Hermite , 4 - GF-Type2 Spherical Cubature

## User Input: For the Filter : For ALL
P_model = 1

Theta_Initial = [ 0.05, 0.05, 10000000, 10000000, 0.5, 0.5]  # [R_zw, R_wa, C_z, C_w, A_1, A_2]

# State/Parameter/Output/Input Size
n_State = 1  # Dimension of states of the system
n_Parameter = 5  # Dimension of parameters of the system
n_Output = 1  # Dimension of measurements of the system
n_Input = 6  # Dimension of inputs of the system

# Filter process/measurement noise covariances : For ALL
Q_model = 0.001
Q_params = 0.001
R_model = 0.01

# Filter constant parameters : Both UKF and GF
n = 8  # Dimension of states of the system - Incudes both States and Parameters
m = 1  # Dimension of measurements of the system

p = 3  # Order of Hermite Polynomial :  For GF

alpha = 0.5 # Controls spread of Sigma Points about mean :  For UKF
k = 0.5  # Controls spread of Sigma Points about mean :  For UKF
beta = 3  # Helps to incorporate prior information about the non-Gaussian :  For UKF

# Plot Size parameters
Plot_Width = 15
Plot_Height = 10

## Data Source dependent User Inputs
if (Type_Data_Source  == 1):  # PNNL Prototype Data

    Simulation_Name = "test1"

    Total_Aggregation_Zone_Number = 5

    ## User Input: Aggregation Unit Number ##
    # Aggregation_UnitNumber = 1
    # Aggregation_UnitNumber = 2

    # Aggregation Zone NameStem Input
    Aggregation_Zone_NameStem = 'Aggregation_Zone'

    ## Providing Proper Extensions depending on Type of Filter/Smoother Utilized and IMPROVEMENT is needed wrt type of data and model type (NINAD'S WORK)
    if (StateAug_Filter == 1):
        GBModel_Key = 'StateAug_EKFS_DS_B' + '_SSM_' + str(Type_System_Model)
    elif (StateAug_Filter == 2):
        GBModel_Key = 'StateAug_UKFS_DS_B' + '_SSM_' + str(Type_System_Model)
    elif (StateAug_Filter == 3):
        GBModel_Key = 'StateAug_GFS1_DS_B' + '_SSM_' + str(Type_System_Model)
    elif (StateAug_Filter == 4):
        GBModel_Key = 'StateAug_GFS2_DS_B' + '_SSM_' + str(Type_System_Model)

    ## Providing Parameter Details for storing Results based on Type of Model
    if (Type_System_Model == 1):  # 1-State

        State_Names = [r'$T_{z}$']

        Theta_Names = ['R_za' , 'C_z' , 'Asol']

    elif (Type_System_Model == 2):  # 2-State

        State_Names = [r'$T_{z}$', r'$T_{w}$']

        Theta_Names = ['R_zw', 'R_wa', 'C_z', 'C_w', 'A1', 'A2']

elif (Type_Data_Source  == 2):  # Simulated 4 State House Thermal Model Data

    Simulated_HouseThermalData_Filename = "HouseThermalData_Gainesville_Baseline_3Months_PVBat1_Bat0_PV0_None0_DC.mat"

    Total_Aggregation_Zone_Number = 1  # Always 1

    Theta_True = [ 0.05, 0.05, 10000000, 10000000, 0.5, 0.5]

    ## Providing Proper Extensions depending on Type of Filter/Smoother Utilized and IMPROVEMENT is needed wrt type of data and model type (NINAD'S WORK)
    if (StateAug_Filter == 1):
        GBModel_Key = 'StateAug_EKFS_DS_H' + '_SSM_' + str(Type_System_Model)
    elif (StateAug_Filter == 2):
        GBModel_Key = 'StateAug_UKFS_DS_H' + '_SSM_' + str(Type_System_Model)
    elif (StateAug_Filter == 3):
        GBModel_Key = 'StateAug_GFS1_DS_H' + '_SSM_' + str(Type_System_Model)
    elif (StateAug_Filter == 4):
        GBModel_Key = 'StateAug_GFS2_DS_H' + '_SSM_' + str(Type_System_Model)

    ## Providing Parameter Details for storing Results based on Type of Model
    if (Type_System_Model == 1):  # 1-State

        State_Names = [r'$T_{ave}$']

        Theta_Names = ['R_win', 'C_in', 'C1', 'C2', 'C3']

    elif (Type_System_Model == 2):  # 2-State

        State_Names = [r'$T_{ave}$', r'$T_{wall}$']

        Theta_Names = ['R_win', 'R_w', 'C_in', 'C_w', 'C1', 'C2', 'C3']

    elif (Type_System_Model == 4):  # 4-State

        State_Names = [r'$T_{ave}$', r'$T_{wall}$', r'$T_{attic}$', r'$T_{im}$']

        Theta_Names = ['R_win', 'R_w', 'R_attic', 'R_im', 'R_roof', 'C_in', 'C_w', 'C_attic', 'C_im', 'C1', 'C2', 'C3']

# Creating Training Folder Name
Training_FolderName = GBModel_Key

## Basic Computation

# Computing ts in seconds
ts = FileResolution_Minutes*60


#--------------------------------------------------------------Defining System Model---------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
if (Type_Data_Source  == 1):  # PNNL Prototype Data

    if (Type_System_Model == 1):  # One State Model Make this from the two state model Kunal

        # Defining the One-State Discrete-Time Nonlinear System Function
        def SingleZone_f1(x_k_1, u_k_1):
            """Provides discrete time dynamics for a nonlinear Single-Zone system
            
            Args:
                x_k_1 (numpy.array): Previous system state
                u_k_1 (numpy.array): Previous system input
                
            Returns:
                x_k_true (numpy.array): Current true system state      
            """
            
            # Simulation parameters
            # ts = 300  # Time step in seconds
            
            # Getting individual state component
            Tz_k_1 = x_k_1[0, 0]  # Previous zone temperature

            # Getting the parameter components : [R_za, C_z, A_sol] 
            Rza_k_1 = x_k_1[1, 0]
            Cz_k_1 = x_k_1[2, 0]
            Asol_k_1 = x_k_1[3, 0]

            # Getting Control Components : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
            Ta = u_k_1[0,0]
            Q_sol1 = u_k_1[1,0]
            Q_sol2 = u_k_1[2,0]
            Q_Zic = u_k_1[3,0]
            Q_Zir = u_k_1[4,0]
            Q_ac = u_k_1[5,0]
            
            # Computing current true system state
            Tz_k = Tz_k_1 + ts * ((1 / (Rza_k_1 * Cz_k_1)) * (Ta - Tz_k_1) + 
                                  (1 / Cz_k_1) * (Q_Zic + Q_Zir + Q_ac) +
                                  (Asol_k_1 / Cz_k_1) * (Q_sol1 + Q_sol2))
            
            # Computing current true system parameters
            Rza_k = Rza_k_1
            Cz_k = Cz_k_1
            Asol_k = Asol_k_1
                                                
            # Return statement
            x_k_true = np.reshape(np.array([Tz_k, Rza_k, Cz_k, Asol_k]), (4, 1))

            return x_k_true

         # One-State Discrete-Time Nonlinear Measurement Function
        def SingleZone_h1(x_k_1):
            """Provides discrete time Measurement for a nonlinear Single-Zone system
            
            Args:
                x_k_1 (numpy.array): Previous system state
                
            Returns:
                y_k_true (numpy.array): Current true system measurement       
            """
            # Simulation parameters
            
            # Getting individual state components
            Tz_k_1 = x_k_1[0,0]
            
            # Computing current true system measurement
            y_k_true = Tz_k_1 
            y_k_true = np.reshape(y_k_true,(1,1))
                                                
            # Return statement
            return y_k_true 

        # One-State Discrete-Time Linear System Jacobian Function
        def SingleZone_F(x_k_1, u_k_1):
            """Provides discrete time nonlinear Single-Zone system Jacobian
            
            Args:
                x_k_1 (numpy.array): Previous system state
                u_k_1 (numpy.array): System input/control
                
            Returns:
                F (numpy.array): System linearized Dynamics Jacobian      
            """
            # Getting individual state component
            Tz_k_1 = x_k_1[0, 0]  # Previous zone temperature

            # Getting the parameter components : [R_za, C_z, A_sol] 
            Rza_k_1 = x_k_1[1, 0]
            Cz_k_1 = x_k_1[2, 0]
            Asol_k_1 = x_k_1[3, 0]

            # Getting Control Components : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
            Ta = u_k_1[0,0]
            Q_sol1 = u_k_1[1,0]
            Q_sol2 = u_k_1[2,0]
            Q_Zic = u_k_1[3,0]
            Q_Zir = u_k_1[4,0]
            Q_ac = u_k_1[5,0]
            
            # Time step
            # ts = 300  # Time step in seconds

            # Computing System State Jacobian
            F = np.reshape(np.array([[1-ts*(1 / (Rza_k_1 * Cz_k_1)), ts*(-1 / (Rza_k_1**2 * Cz_k_1)) * (Ta - Tz_k_1), ts * (-1/Cz_k_1**2)((1 / (Rza_k_1)) * (Ta - Tz_k_1) + (Q_Zic + Q_Zir + Q_ac) + (Asol_k_1) * (Q_sol1 + Q_sol2)), ts*((1/ Cz_k_1) * (Q_sol1 + Q_sol2))], 
                                    [0,1,0,0], 
                                    [0,0,1,0],
                                    [0,0,0,1]]),(4,4))
            # Rest of the F matrix remains zero as parameters are assumed constant over one time step
            
            return F

        # One-State Discrete-Time Measurement Jacobian Function
        def SingleZone_H(x_k_1):
            """Provides discrete time linearized measurement Jacobian for a Single-Zone system
            
            Args:
                x_k_1 (numpy.array): Current system state
                
            Returns:
                H (numpy.array): Linearized measurement Jacobian
            """
            # Computing System State Jacobian
            H = np.reshape(np.array([1, 0, 0, 0]),(1,4))
            
                                                
            # Return statement
            return H 

    elif (Type_System_Model == 2):  # Two State Model

        # Defining the Simple Pendulum Discrete-Time Nonlinear System Function
        def SingleZone_f1(x_k_1, u_k_1):
            """Provides discrete time dynamics for a nonlinear Single-Zone system
            
            Args:
                x_k_1 (numpy.array): Previous system state
                u_k_1 (numpy.array): Previous system measurement
                
            Returns:
                x_k_true (numpy.array): Current true system state      
            """
            
            # Simulation parameters
            # ts = 300    
            
            # Getting individual state components
            Tz_k_1 = x_k_1[0,0]
            Tw_k_1 = x_k_1[1,0]

            # Getting the parameter components : [R_zw, R_wa, C_z, C_w, A_1, A_2] 
            Rzw_k_1 = x_k_1[2,0]
            Rwa_k_1 = x_k_1[3,0]
            Cz_k_1 = x_k_1[4,0]
            Cw_k_1 = x_k_1[5,0]
            A1_k_1 = x_k_1[6,0]
            A2_k_1 = x_k_1[7,0]

            # Getting Control Components : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
            Ta = u_k_1[0,0]
            Q_sol1 = u_k_1[1,0]
            Q_sol2 = u_k_1[2,0]
            Q_Zic = u_k_1[3,0]
            Q_Zir = u_k_1[4,0]
            Q_ac = u_k_1[5,0]
            
            # Computing current true system state
            Tz_k = Tz_k_1 + ts*(((1/(Cz_k_1 * Rzw_k_1)) * (Tw_k_1 - Tz_k_1)) + 
                                ((1/Cz_k_1) * (Q_Zic + Q_ac)) +
                                ((A1_k_1/Cz_k_1) * (Q_sol1)))
            Tw_k = Tw_k_1 + ts*(((1/(Cw_k_1 * Rzw_k_1)) * (Tz_k_1 - Tw_k_1)) +
                                ((1/(Cw_k_1 * Rwa_k_1)) * (Ta - Tw_k_1)) +
                                ((1/Cw_k_1) * (Q_Zir)) +
                                ((A2_k_1/Cw_k_1) * (Q_sol2))) 

            # Computing current true system parameters
            Rzw_k = Rzw_k_1
            Rwa_k = Rwa_k_1
            Cz_k = Cz_k_1
            Cw_k = Cw_k_1
            A1_k = A1_k_1
            A2_k = A2_k_1
                                                
            x_k_true = np.reshape(np.array([Tz_k, Tw_k, Rzw_k, Rwa_k, Cz_k, Cw_k, A1_k, A2_k]),(8,1))
                                                
            # Return statement
            return x_k_true 

        # Defining the Simple Pendulum Discrete-Time Nonlinear Measurement Function
        def SingleZone_h1(x_k_1):
            """Provides discrete time Measurement for a nonlinear Single-Zone system
            
            Args:
                x_k_1 (numpy.array): Previous system state
                
            Returns:
                y_k_true (numpy.array): Current true system measurement       
            """
            
            # Simulation parameters
            
            # Getting individual state components
            Tz_k_1 = x_k_1[0,0]
            
            # Computing current true system measurement
            y_k_true = Tz_k_1 
            y_k_true = np.reshape(y_k_true,(1,1))
                                                
            # Return statement
            return y_k_true 


        #----------------------------------------------------------Defining System Model Jacobian----------------------------------------------------------------#
        #--------------------------------------------------------------------------------------------------------------------------------------------------------#
        # Defining the Single-Zone Discrete-Time Linear System Function
        def SingleZone_F(x_k_1, u_k_1):
            """Provides discrete time nonlinear Single-Zone system Jacobian
            
            Args:
                x_k_1 (numpy.array): Previous system state
                u_k_1 (numpy.array): Previous system measurement
                
            Returns:
                F (numpy.array): System linearized Dynamics Jacobian      
            """
                
            # Getting individual state components
            Tz_k_1 = x_k_1[0,0]
            Tw_k_1 = x_k_1[1,0]

            # Getting the parameter components
            Rzw_k_1 = x_k_1[2,0]
            Rwa_k_1 = x_k_1[3,0]
            Cz_k_1 = x_k_1[4,0]
            Cw_k_1 = x_k_1[5,0]
            A1_k_1 = x_k_1[6,0]
            A2_k_1 = x_k_1[7,0]

            # Getting Control Components : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
            Ta = u_k_1[0,0]
            Q_sol1 = u_k_1[1,0]
            Q_sol2 = u_k_1[2,0]
            Q_Zic = u_k_1[3,0]
            Q_Zir = u_k_1[4,0]
            Q_ac = u_k_1[5,0]
            
            # Computing System State Jacobian
            F = np.reshape(np.array([[1-ts*(1/(Cz_k_1 * Rzw_k_1)), ts*(1/(Cz_k_1 * Rzw_k_1)), ts*(-(1/(Cz_k_1 * Rzw_k_1**2)) * (Tw_k_1 - Tz_k_1)), 0, ts*(-(1/Cz_k_1**2) * (((Tw_k_1 - Tz_k_1)/(Rzw_k_1)) + (Q_Zic + Q_ac) + (A1_k_1 * Q_sol1))), 0, ts*(Q_sol1/Cz_k_1), 0], 
                                    [ts*(1/(Cw_k_1 * Rzw_k_1)), 1-ts*((1/(Cw_k_1 * Rzw_k_1)) +(1/(Cw_k_1 * Rwa_k_1))), ts*(-(1/(Cw_k_1 * Rzw_k_1**2)) * (Tz_k_1 - Tw_k_1)), ts*(-(1/(Cw_k_1 * Rwa_k_1**2)) * (Ta - Tw_k_1)), 0, ts*(-(1/Cw_k_1**2) * (((Tz_k_1 - Tw_k_1)/(Rzw_k_1)) + ((Ta - Tw_k_1)/(Rwa_k_1)) + (Q_Zir) + (A2_k_1 * Q_sol2))), 0, ts*(Q_sol2/Cw_k_1)], 
                                    [0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 1]]),(8,8))
            
                                                
            # Return statement
            return F 

        # Defining the Simple Pendulum Discrete-Time Linearized Measurement Function
        def SingleZone_H(x_k_1):
            """Provides discrete time nonlinear Single-Zone system Measurement Jacobian
            
            Args:
                x_k_1 (numpy.array): Previous system state
                
            Returns:
                H (numpy.array): System linearized Measurement Jacobian      
            """
                
            
            
            # Computing System State Jacobian
            H = np.reshape(np.array([1, 0, 0, 0, 0, 0, 0, 0]),(1,8))
            
                                                
            # Return statement
            return H     

elif (Type_Data_Source  == 2):  # Simulated 4 State House Thermal Model Data
 
    if (Type_System_Model == 1):  # Two State Model
 
        # Defining the Simple Pendulum Discrete-Time Nonlinear System Function
        def SingleZone_f1(x_k, u_k):
            """Provides discrete time dynamics for a nonlinear Single-Zone system
           
            Args:
                x_k (numpy.array): Previous system state
                u_k (numpy.array): Previous system measurement
               
            Returns:
                x_k_true (numpy.array): Current true system state      
            """
           
            # Simulation parameters
            #ts = 300    
           
            # Getting individual state components
            Tave_k = x_k[0,0]
 
            Rwin_k = x_k[1,0]
            Cin_k = x_k[2,0]
            C1_k = x_k[3,0]
            C2_k = x_k[4,0]
            C3_k = x_k[5,0]
 
            Tam = u_k[0,0]
            Q_ih = u_k[1, 0]
            Q_ac = u_k[2,0]
            Q_venti = u_k[3,0]
            Q_infil = u_k[4,0]
            Q_solar = u_k[5,0]
       
            # Computing current true system state
            Tave_k_1 = Tave_k + ts * (1 / (Cin_k)) * (((Tam - Tave_k) / Rwin_k) + C1_k * Q_ih + C2_k * Q_ac + C3_k * Q_solar + Q_venti + Q_infil)
 
 
            # Computing current true system parameters
            Rwin_k_1 = Rwin_k
            Cin_k_1 = Cin_k
            C1_k_1 = C1_k
            C2_k_1 = C2_k
            C3_k_1 = C3_k
                                               
            x_k_true = np.reshape(np.array([Tave_k_1, Rwin_k_1 , Cin_k_1, C1_k_1, C2_k_1, C3_k_1], (6,1)))
                                               
            # Return statement
            return x_k_true
 
        # Defining the Simple Pendulum Discrete-Time Nonlinear Measurement Function
        def SingleZone_h1(x_k):
            """Provides discrete time Measurement for a nonlinear Single-Zone system
           
            Args:
                x_k (numpy.array): Previous system state
               
            Returns:
                y_k_true (numpy.array): Current true system measurement      
            """
           
            # Simulation parameters
           
            # Getting individual state components
            Tz_k_1 = x_k[0,0]
           
            # Computing current true system measurement
            y_k_true = Tz_k_1
            y_k_true = np.reshape(y_k_true,(1,1))
                                               
            # Return statement
            return y_k_true
 
        def SingleZone_F(x_k, u_k):
            """
            Compute the Jacobian matrix of partial derivatives for the system state and parameters.
 
            Args:
            x_k (numpy.array): Current state and parameters [Tave_k, Cin, C1, C2, C3, Rwin]
            u_k (numpy.array): Current inputs [Tam, Qih, Qac, Qsolar, Qventi, Qinfil]
 
            Returns:
            numpy.array: Jacobian matrix of size 6x6.
            """
            Tave_k = x_k[0,0]
 
            Rwin_k = x_k[1,0]
            Cin_k= x_k[2,0]
            C1_k=x_k[3,0]
            C2_k=x_k[4,0]
            C3_k=x_k[5,0]
 
            Tam = u_k[0,0]
            Q_ih = u_k[1, 0]
            Q_ac = u_k[2,0]
            Q_venti = u_k[3,0]
            Q_infil = u_k[4,0]
            Q_solar = u_k[5,0]
 
           
            # Initialize the Jacobian matrix with zeros
            F = np.zeros((6, 6))  # 1 state and 5 parameters
 
 
            F = np.reshape(np.array([[ 1 - ts / (Cin_k * Rwin_k),-( ts * (Tam - Tave_k) / (Cin_k * Rwin_k**2) ), -ts * (Tam - Tave_k) / (Cin_k**2 * Rwin_k), ts *( Q_ih/Cin_k) , ts * (Q_ac/Cin_k),  ts * (Q_solar/Cin_k)]
                                            [0,1,0,0,0,0],
                                            [0,0,1,0,0,0],
                                            [0,0,0,1,0,0]
                                            [0,0,0,0,1,0]
                                            [0,0,0,0,0,1]]),(6,6))
            return F
 
        # Defining the Simple Pendulum Discrete-Time Linearized Measurement Function
        def SingleZone_H(x_k):
            """Provides discrete time nonlinear Single-Zone system Measurement Jacobian
           
            Args:
                x_k (numpy.array): Previous system state
               
            Returns:
                H (numpy.array): System linearized Measurement Jacobian      
            """
                           
            # Computing System State Jacobian
            H = np.reshape(np.array([1, 0, 0, 0, 0, 0]),(1,6))
           
                                               
            # Return statement
            return H
 
    elif (Type_System_Model == 2):  # Two State Model
 
        # Defining the Simple Pendulum Discrete-Time Nonlinear System Function
        def SingleZone_f1(x_k, u_k):
            """Provides discrete time dynamics for a nonlinear Single-Zone system
           
            Args:
                x_k (numpy.array): Previous system state
                u_k (numpy.array): Previous system measurement
               
            Returns:
                x_k_true (numpy.array): Current true system state      
            """
           
            # Simulation parameters
            #ts = 300    
           
            # Getting individual state components
            Tave_k = x_k[0,0]
            Twall_k = x_k[1,0]
 
            # Getting the parameter components :
            Rw_k = x_k[2,0]
            Rwin_k = x_k[3,0]
            Cw_k = x_k[4,0]
            Cin_k = x_k[5,0]
            C1_k = x_k[6,0]
            C2_k = x_k[7,0]
            C3_k = x_k[8,0]
 
            # Getting Control Component/inputs : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
            Tsolw_k = u_k[0,0]
            Tam = u_k[1,0]
            Q_ih = u_k[2, 0]
            Q_ac = u_k[3,0]
            Q_venti = u_k[4,0]
            Q_infil = u_k[5,0]
            Q_solar = u_k[6,0]
           
           
            # Computing current true system state
            Tave_k_1 = Tave_k + ts * ((1/Cin_k) * ((((Twall_k-Tave_k)/Rw_k)*2) + (((Tam-Tave_k)/Rwin_k))+ (C1_k*Q_ih) + (C2_k * Q_ac) + Q_venti + Q_infil))
            Twall_k_1 = Twall_k + ts * ((1/Cw_k) * ((((Tave_k-Twall_k)/Rw_k)*2) + (((Tsolw_k-Twall_k)/Rw_k)*2)+ (C3_k*Q_solar)))            
 
            # Computing current true system parameters
            Rw_k_1 = Rw_k
            Rwin_k_1 = Rwin_k
            Cw_k_1 = Cw_k
            Cin_k_1 = Cin_k
            C1_k_1 = C1_k
            C2_k_1 = C2_k
            C3_k_1 = C3_k
                                               
            x_k_true = np.reshape(np.array([Tave_k_1, Twall_k_1,  Rw_k_1 , Rwin_k_1 , Cw_k_1 , Cin_k_1, C1_k_1, C2_k_1, C3_k_1], (9,1)))
                                               
            # Return statement
            return x_k_true
 
                # Defining the Simple Pendulum Discrete-Time Nonlinear Measurement Function
        def SingleZone_h1(x_k):
            """Provides discrete time Measurement for a nonlinear Single-Zone system
           
            Args:
                x_k (numpy.array): Previous system state
               
            Returns:
                y_k_true (numpy.array): Current true system measurement      
            """
           
            # Simulation parameters
           
            # Getting individual state components
            Tz_k_1 = x_k[0,0]
           
            # Computing current true system measurement
            y_k_true = Tz_k_1
            y_k_true = np.reshape(y_k_true,(1,1))
                                               
            # Return statement
            return y_k_true
 
 
 
        def SingleZone_F(x_k, u_k):
            """
            Compute the Jacobian matrix of partial derivatives for the system state and parameters.
 
            Args:
            x_k (numpy.array): Current state and parameters [Tave_k, Twall_k, Rw_k, ..., C3_k]
            u_k (numpy.array): Current inputs [Tsolw, Tam, Q_ac, Q_venti, Q_infil, Q_solar, Q_ih]
 
            Returns:
            numpy.array: Jacobian matrix of size 11x11.
            """
            # Getting individual state components
            Tave_k = x_k[0,0]
            Twall_k = x_k[1,0]
 
            # Getting the parameter components :
            Rw_k = x_k[2,0]
            Rwin_k = x_k[3,0]
            Cw_k= x_k[4,0]
            Cin_k= x_k[5,0]
            C1_k=x_k[6,0]
            C2_k=x_k[7,0]
            C3_k=x_k[8,0]
 
            # Getting Control Component/inputs : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
            Tsolw_k = u_k[0,0]
            Tam = u_k[1,0]
            Q_ih = u_k[2, 0]
            Q_ac = u_k[3,0]
            Q_venti = u_k[4,0]
            Q_infil = u_k[5,0]
            Q_solar = u_k[6,0]
 
            # Initialize the Jacobian matrix with zeros
            F = np.zeros((11, 11))
           
 
            F = np.reshape(np.array( [1- ts*((1/(Cin_k* (Rw_k/2)))+ (1/(Cin_k*Rwin_k))), -ts*(1/(Cin_k*(Rw_k/2))) , -ts * ((1/Cin_k)* ((Twall_k-Tave_k)/0.5)) * (1/(Rw_k*Rw_k)),  -ts * ((1/Cin_k)* ((Tam-Tave_k)/(Rwin_k*Rwin_k))), 0,
                                            -ts * ((2 * ((Twall_k-Tave_k) / (Rw_k)) + ((Tam -Tave_k)/(Rwin_k)) + (C1_k*Q_ih) + (C2_k * Q_ac) + Q_venti + Q_infil)*(1/ (Cin_k*Cin_k))), ts* ((1/Cin_k) * Q_ih),
                                            ts* ((1/Cin_k) * Q_ac), 0 ],
                                            [ts* (1/ (Cw_k * ((Rw_k)/2))), 1-ts * (4/ (Cw_k * Rw_k)), -ts * ((1/Cw_k)* (((Tave_k -Twall_k)/ 0.5)* (1/(Rw_k* Rw_k)) + ((Tsolw_k -Twall_k)/ 0.5)* (1/(Rw_k* Rw_k)))),
                                            0, -ts * (((Tave_k -Twall_k)/ (Rw_k/2)) + ((Tsolw_k -Twall_k)/ (Rw_k/2)) + C3_k * Q_solar) * (1/(Cw_k* Cw_k)), 0, 0, 0, ts* ((1/Cw_k) * Q_solar)] ,
                                                    [0,0,1,0,0,0,0,0,0],
                                                    [0,0,0,1,0,0,0,0,0],
                                                    [0,0,0,0,1,0,0,0,0],
                                                    [0,0,0,0,0,1,0,0,0],
                                                    [0,0,0,0,0,0,1,0,0],
                                                    [0,0,0,0,0,0,0,1,0],                                    
                                                    [0,0,0,0,0,0,0,0,1]),(9,9))
               
 
            return F
 
                # Defining the Simple Pendulum Discrete-Time Linearized Measurement Function
        def SingleZone_H(x_k):
            """Provides discrete time nonlinear Single-Zone system Measurement Jacobian
           
            Args:
                x_k (numpy.array): Previous system state
               
            Returns:
                H (numpy.array): System linearized Measurement Jacobian      
            """
               
            # Getting individual state components
            theta_k_1 = x_k[0,0]
            omega_k_1 = x_k[1,0]
           
            # Computing System State Jacobian
            H = np.reshape(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),(1,9))
           
                                               
            # Return statement
            return H
   
    elif (Type_System_Model == 4):  # Four State Model
 
        # Defining the Four-State Discrete-Time Nonlinear System Function
        def SingleZone_f1(x_k, u_k):
            """Provides discrete time dynamics for a nonlinear Four-State system
            Args:
                x_k (numpy.array): Previous system state
                u_k (numpy.array): System input/control
            Returns:
                x_k (numpy.array): Current system state
            """
            # Unpack state components
            T_ave_k = x_k[0, 0]
            T_wall_k=  x_k[1, 0]            
            T_attic_k =  x_k[2, 0]
            T_im_k = x_k[3, 0]
 
            # parameters assuming they are part of the state vector)
            Rw = x_k[4, 0]
            Rwin = x_k[5, 0]
            Rattic = x_k[6, 0]
            Rroof = x_k[7, 0]
            Rim = x_k[8, 0]
            Cw = x_k[9, 0]
            Cin = x_k[10, 0]
            Cattic = x_k[11, 0]
            Cim = x_k[12, 0]            
            C1 = x_k[13, 0]
            C2 = x_k[14, 0]
            C3 = x_k[15, 0]
           
            # Unpack input controls and disturbances
            Tsolw_k = u_k[0,0]
            Tsolr_k = u_k[1,0]
            Tam = u_k[2,0]
            QIHL = u_k[3,0]
            QAC = u_k[4,0]
            Qventi = u_k[5,0]
            Qinfil = u_k[6,0]
            Qsolar = u_k[7,0]
 
            # Define the system equations
            T_ave_k_1= T_ave_k + ts * ((T_wall_k - T_ave_k) / ((Cin * Rw)/2) + (T_attic_k - T_ave_k) / (Cin * Rattic) +
                                    (T_im_k - T_ave_k) / (Cin * Rim) + (Tam - T_ave_k) / (Cin * Rwin) + C1*QIHL + C2 * QAC + Qventi + Qinfil)
            T_wall_k_1 = T_wall_k + ts * ((Tsolw_k - T_wall_k) / ((Cw * Rw)/2) - (T_wall_k - T_ave_k) / ((Cw * Rw)/2))
            T_attic_k_1 = T_attic_k + ts * ((Tsolr_k - T_attic_k) / (Cattic * Rroof) - (T_attic_k - T_ave_k) / (Cattic * Rattic))                      
            T_im_k_1= T_im_k + ts * ((T_im_k - T_ave_k ) / (Cim * Rim) + C3 * Qsolar)

            Rw_k_1 = Rw
            Rwin_k_1 = Rwin
            Rattic_k_1 = Rattic
            Rroof_k_1 = Rroof
            Rim_k_1 = Rim
            Cw_k_1 = Cw
            Cin_k_1 =Cin
            Cattic_k_1 = Cattic
            Cim_k_1 = Cim            
            C1_k_1 =C1
            C2_k_1 = C2
            C3_k_1 = C3
 
            # Assuming the parameters do not change over one time step
            params_k = x_k[4:]
 
            # Combine the state and parameters for the next time step
            x_k_true = np.reshape(np.array([T_ave_k_1, T_wall_k_1,  T_attic_k_1 ,T_im_k_1, Rw_k_1, Rwin_k_1, Rattic_k_1, Rroof_k_1, Rim_k_1, Cw_k_1 , Cin_k_1,  Cattic_k_1,  Cim_k_1 , C1_k_1, C2_k_1, C3_k_1], (16,1)))
     
            return x_k_true
 
        # Defining the Four-State Discrete-Time Nonlinear Measurement Function
        def SingleZone_h1(x_k):
            """Provides discrete time Measurement for a nonlinear Four-State system
            Args:
                x_k (numpy.array): Current system state
            Returns:
                y_k (numpy.array): Current system measurement
            """
            # Direct measurement of the average air temperature for simplicity
            y_k = np.array([x_k[0, 0]])  # Here we're assuming that T_ave is being measured
           
            return y_k.reshape((1, 1))
 
        def SingleZone_F(x_k, u_k):
            """Provides discrete time linearized system dynamics Jacobian for a Four-State system
            Args:
                x_k (numpy.array): Previous system state
                u_k (numpy.array): System input/control
            Returns:
                F (numpy.array): Linearized system dynamics Jacobian
            """
    
            # Unpack state components
            T_ave_k = x_k[0, 0]
            T_wall_k=  x_k[1, 0]            
            T_attic_k =  x_k[2, 0]
            T_im_k = x_k[3, 0]
    
            # parameters assuming they are part of the state vector)
            Rw=x_k[4, 0]
            Rwin= x_k[5, 0]
            Rattic= x_k[6, 0]
            Rroof= x_k[7, 0]
            Rim= x_k[8, 0]
            Cw = x_k[9, 0]
            Cin= x_k[10, 0]
            Cattic= x_k[11, 0]
            Cim= x_k[12, 0]            
            C1= x_k[13, 0]
            C2= x_k[14, 0]
            C3= x_k[15, 0]
        
            # Unpack input controls and disturbances
            Tsolw_k = u_k[0,0]
            Tsolr_k = u_k[1,0]
            Tam_k = u_k[2,0]
            QIHL_k = u_k[3,0]
            QAC_k = u_k[4,0]
            Qventi_k = u_k[5,0]
            Qinfil_k = u_k[6,0]
            Qsolar_k = u_k[7,0]    
    
    
            # Initialize the Jacobian matrix with zeros
            F = np.zeros((16, 16))                    
    
            F = np.reshape(np.array(
            [[1+ ts * (-1/(Cin*(Rw/2)))-(1/(Cin*(Rattic)))-(1/(Cin*Rim))-(1/(Cin*(Rwin/2))) , ts* (1/ (Cin * (Rw/2))), - ts*(((T_attic_k-T_ave_k)/ (Cin*Rattic)) * (1/(Rattic * Rattic))), ts * (1 / (Cin * Rim)),
            -ts * ((( T_wall_k-T_ave_k)/ (Cin/2)) * (1/ (Rw*Rw))), -ts* (((Tam_k-T_ave_k)/Cin)* (1/ (Rwin * Rwin))), -ts* (((T_attic_k-T_ave_k) / Cin)* (1/ (Rattic * Rattic))), 0,
            -ts* ((T_im_k-T_ave_k)/ (Cin * (1/(Rim*Rim)))), 0, -ts* ((1/(Cin*Cin)) *(((T_wall_k-T_ave_k) / (Rw/2)) + ((T_attic_k-T_ave_k) / Rattic) + ((T_im_k-T_ave_k) / Rim) +
            ((Tam_k-T_ave_k) / Rwin))), 0, 0, (QIHL_k)*ts, (Qventi_k)* ts, (Qinfil_k) * ts],

            [-ts*(1/(Cw*(Rw/2))), 1, 0, 0, -ts * ((((Tsolw_k-T_wall_k)/(Cw/2))* (1/(Rw*Rw))) + (((T_wall_k - T_ave_k)/ (Cw/2))*(1/(Rw*Rw)))), 0, 0, 0, 0,
            -ts * ((((Tsolw_k-T_wall_k)/(Rw/2))* (1/(Cw*Cw))) + (((T_wall_k - T_ave_k)/ (Rw/2))*(1/(Cw*Cw)))), 0, 0, 0, 0, 0, 0],

            [ ts* (1/Cattic*Rattic), 0, 1- ts* (1/(Cattic*Rattic)), 0, 0, 0, ts * (((T_attic_k-T_ave_k)/Cattic) * (1/(Rattic*Rattic))), ts * -(((Tsolr_k-T_attic_k)/Cattic) * (1/(Rroof*Rroof))),
            0, 0, 0, -ts * ((((Tsolr_k-T_attic_k)/(Rroof)) * (1/(Cattic*Cattic))) +   ((T_attic_k-T_ave_k)/Rattic * 1(1/(Cattic*Cattic)))), 0, 0, 0, 0]

            [-ts/ (Cim*Rim), 0, 0, 1+ts* (1/ (Cim*Rim)), 0, 0, 0, 0, -ts*(((T_im_k-T_ave_k)/(Cim)) * (1/(Rim*Rim))), 0, 0, 0, -ts* (((T_im_k-T_ave_k)/(Rim))* (1/ (Cim*Cim))), 0, 0, Qsolar_k],
            [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
            [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]),(16,16))
    
    
            return F
    
        # Defining the Four-State Discrete-Time Measurement Jacobian Function
        def SingleZone_H(x_k):
            """Provides discrete time linearized measurement Jacobian for a Four-State system
            Args:
                x_k (numpy.array): Current system state
            Returns:
                H (numpy.array): Linearized measurement Jacobian
            """
            H = np.zeros((1, 16))
            H[0, 0] = 1  # Assuming the second state, T_ave, is directly measured
    
            return H
        

#--------------------------------------------------------------Defining System Model for Simulation---------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def Building_GreyBox_Simulator(x_k, u_k, Parameter_Vector, Type_Data_Source, Type_System_Model):        
    """Provides discrete time linearized system dynamics Jacobian for a Four-State system
        Args:
            x_k (numpy.array): Previous system state
            u_k (numpy.array): System input/control
            Parameter_Vector (numpy.array): System Parameters
            Type_Data_Source (Int): Type of data either EnergyPlus or from 4th order house model
            Type_System_Model (Int): Order of RC-Network model
        Returns:
            x_k (numpy.array): Current system state
    """       

    # IF Else Loop: Based on Type_Data_Source
    if (Type_Data_Source == 1):  # PNNL Prototype Data

        # IF Else Loop: Based on Type_System_Model
        if (Type_System_Model == 1):  # 1-State RC-Network

            # Getting individual state component
            Tz_k_1 = x_k[0, 0]  # Previous zone temperature

            # Getting the parameter components : [R_za, C_z, A_sol] 
            Rza_k_1 = Parameter_Vector[0, 0]
            Cz_k_1 = Parameter_Vector[1, 0]
            Asol_k_1 = Parameter_Vector[2, 0]

            # Getting Control Components : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
            Ta = u_k[0,0]
            Q_sol1 = u_k[1,0]
            Q_sol2 = u_k[2,0]
            Q_Zic = u_k[3,0]
            Q_Zir = u_k[4,0]
            Q_ac = u_k[5,0]
            
            # Computing current true system state
            Tz_k = Tz_k_1 + ts * ((1 / (Rza_k_1 * Cz_k_1)) * (Ta - Tz_k_1) + 
                                  (1 / Cz_k_1) * (Q_Zic + Q_Zir + Q_ac) +
                                  (Asol_k_1 / Cz_k_1) * (Q_sol1 + Q_sol2))           
            
                                                
            # Returning Next State
            x_k_1 = np.reshape(np.array([Tz_k]), (1, 1))

        elif (Type_System_Model == 2):  # 2-State RC-Network

            # Getting individual state components
            Tz_k_1 = x_k[0,0]
            Tw_k_1 = x_k[1,0]

            # Getting the parameter components : [R_zw, R_wa, C_z, C_w, A_1, A_2] 
            Rzw_k_1 = Parameter_Vector[0,0]
            Rwa_k_1 = Parameter_Vector[1,0]
            Cz_k_1 = Parameter_Vector[2,0]
            Cw_k_1 = Parameter_Vector[3,0]
            A1_k_1 = Parameter_Vector[4,0]
            A2_k_1 = Parameter_Vector[5,0]

            # Getting Control Components : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
            Ta = u_k[0,0]
            Q_sol1 = u_k[1,0]
            Q_sol2 = u_k[2,0]
            Q_Zic = u_k[3,0]
            Q_Zir = u_k[4,0]
            Q_ac = u_k[5,0]
            
            # Computing current true system state
            Tz_k = Tz_k_1 + ts*(((1/(Cz_k_1 * Rzw_k_1)) * (Tw_k_1 - Tz_k_1)) + 
                                ((1/Cz_k_1) * (Q_Zic + Q_ac)) +
                                ((A1_k_1/Cz_k_1) * (Q_sol1)))
            Tw_k = Tw_k_1 + ts*(((1/(Cw_k_1 * Rzw_k_1)) * (Tz_k_1 - Tw_k_1)) +
                                ((1/(Cw_k_1 * Rwa_k_1)) * (Ta - Tw_k_1)) +
                                ((1/Cw_k_1) * (Q_Zir)) +
                                ((A2_k_1/Cw_k_1) * (Q_sol2))) 
            
            # Returning Next State
            x_k_1 = np.reshape(np.array([Tz_k, Tw_k]),(2,1))


    elif (Type_Data_Source == 2):  # 4th Order House Thermal Model

        # IF Else Loop: Based on Type_System_Model
        if (Type_System_Model == 1):  # 1-State RC-Network

            # Getting individual state components
            Tave_k = x_k[0,0]
 
            Rwin_k = Parameter_Vector[0,0]
            Cin_k = Parameter_Vector[1,0]
            C1_k = Parameter_Vector[2,0]
            C2_k = Parameter_Vector[3,0]
            C3_k = Parameter_Vector[4,0]
 
            Tam = u_k[0,0]
            Q_ih = u_k[1, 0]
            Q_ac = u_k[2,0]
            Q_venti = u_k[3,0]
            Q_infil = u_k[4,0]
            Q_solar = u_k[5,0]
       
            # Computing current true system state
            Tave_k_1 = Tave_k + ts * (1 / (Cin_k)) * (((Tam - Tave_k) / Rwin_k) + C1_k * Q_ih + C2_k * Q_ac + C3_k * Q_solar + Q_venti + Q_infil) 
 
            # Returning Next State                                   
            x_k_1 = np.reshape(np.array([Tave_k_1], (1,1)))

        elif (Type_System_Model == 2):  # 2-State RC-Network

            # Getting individual state components
            Tave_k = x_k[0,0]
            Twall_k = x_k[1,0]
 
            # Getting the parameter components :
            Rw_k = Parameter_Vector[0,0]
            Rwin_k = Parameter_Vector[1,0]
            Cw_k = Parameter_Vector[2,0]
            Cin_k = Parameter_Vector[3,0]
            C1_k = Parameter_Vector[4,0]
            C2_k = Parameter_Vector[5,0]
            C3_k = Parameter_Vector[6,0]
 
            # Getting Control Component/inputs : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]
            Tsolw_k = u_k[0,0]
            Tam = u_k[1,0]
            Q_ih = u_k[2, 0]
            Q_ac = u_k[3,0]
            Q_venti = u_k[4,0]
            Q_infil = u_k[5,0]
            Q_solar = u_k[6,0]
           
           
            # Computing current true system state
            Tave_k_1 = Tave_k + ts * ((1/Cin_k) * ((((Twall_k-Tave_k)/Rw_k)*2) + (((Tam-Tave_k)/Rwin_k))+ (C1_k*Q_ih) + (C2_k * Q_ac) + Q_venti + Q_infil))
            Twall_k_1 = Twall_k + ts * ((1/Cw_k) * ((((Tave_k-Twall_k)/Rw_k)*2) + (((Tsolw_k-Twall_k)/Rw_k)*2)+ (C3_k*Q_solar)))   
            
            # Returning Next State
            x_k_1 = np.reshape(np.array([Tave_k_1, Twall_k_1], (2,1)))

        elif (Type_System_Model == 4):  # 4-State RC-Network

            # Unpack state components
            T_ave_k = x_k[0, 0]
            T_wall_k=  x_k[1, 0]            
            T_attic_k =  x_k[2, 0]
            T_im_k = x_k[3, 0]
 
            # parameters assuming they are part of the state vector)
            Rw = Parameter_Vector[0, 0]
            Rwin = Parameter_Vector[1, 0]
            Rattic = Parameter_Vector[2, 0]
            Rroof = Parameter_Vector[3, 0]
            Rim = Parameter_Vector[4, 0]
            Cw = Parameter_Vector[5, 0]
            Cin = Parameter_Vector[6, 0]
            Cattic = Parameter_Vector[7, 0]
            Cim = Parameter_Vector[8, 0]            
            C1 = Parameter_Vector[9, 0]
            C2 = Parameter_Vector[10, 0]
            C3 = Parameter_Vector[11, 0]
           
            # Unpack input controls and disturbances
            Tsolw_k = u_k[0,0]
            Tsolr_k = u_k[1,0]
            Tam = u_k[2,0]
            QIHL = u_k[3,0]
            QAC = u_k[4,0]
            Qventi = u_k[5,0]
            Qinfil = u_k[6,0]
            Qsolar = u_k[7,0]
 
            # Define the system equations
            T_ave_k_1= T_ave_k + ts * ((T_wall_k - T_ave_k) / ((Cin * Rw)/2) + (T_attic_k - T_ave_k) / (Cin * Rattic) +
                                    (T_im_k - T_ave_k) / (Cin * Rim) + (Tam - T_ave_k) / (Cin * Rwin) + C1*QIHL + C2 * QAC + Qventi + Qinfil)
            T_wall_k_1 = T_wall_k + ts * ((Tsolw_k - T_wall_k) / ((Cw * Rw)/2) - (T_wall_k - T_ave_k) / ((Cw * Rw)/2))
            T_attic_k_1 = T_attic_k + ts * ((Tsolr_k - T_attic_k) / (Cattic * Rroof) - (T_attic_k - T_ave_k) / (Cattic * Rattic))                      
            T_im_k_1= T_im_k + ts * ((T_im_k - T_ave_k ) / (Cim * Rim) + C3 * Qsolar)
 
            # # Returning Next State
            x_k_true = np.reshape(np.array([T_ave_k_1, T_wall_k_1,  T_attic_k_1 ,T_im_k_1], (4,1)))
     

    # Returning Next State
    return x_k_1

#--------------------------------------------------------------Defining Function for Time Series Simulation---------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def Building_GreyBox_Simulator_Data_Sim(Y_Measurement, U_Measurement, Q_Measurement, X_Initial, Parameter_Vector, Type_Data_Source, Type_System_Model, Type_Simulation):
    """Provides discrete time linearized system dynamics Jacobian for a Four-State system
        Args:
            Y_Measurement (numpy.array): System output measurement data
            U_Measurement (numpy.array): System input/control measurement data
            Q_Measurement (numpy.array): System M_Dot and Ts for Type 1 Data Source
            X_Initial (numpy.array): System initial condition of states
            Parameter_Vector (numpy.array): System Parameters
            Type_Data_Source (Int): Type of data either EnergyPlus or from 4th order house model
            Type_System_Model (Int): Order of RC-Network model
            Type_Simulation (Int): Type of Simjulation either without or with state feedback
        Returns:
            X (numpy.array): System State Evolution
            PHVAC (numpy.array): HVAC Power consumed by the building
    """  

    # Constant for Thermal Capacity of Air
    Ca = 1.004

    # Getting number of Iterations
    N_iter = Y_Measurement.shape[1]

    # Setting up State Vector
    X = X_Initial

    X_State = X

    # Initializing
    PHVAC_Current = np.zeros((N_iter,1))

    # FOR LOOP: For Simulation
    for ii in range(N_iter):

        # If Else Loop: Depending on Type of Data Source - Computing PHVAC
        if (Type_Data_Source == 1):  # Energyplus PNNL Prototype Data

            # Getting Current Tz, M_Dot, and Ts
            Tz_Current = X[0,ii]
            M_Dot_Current = Q_Measurement[0,ii]
            Ts_Current = Q_Measurement[1,ii]

            # Computing QHVAC_Current
            QHVAC_Current = Ca * M_Dot_Current * (Tz_Current - Ts_Current)

            # Computing PHVAC
            PHVAC_Current[ii,0] = PHVAC_Regression_Model(np.abs(QHVAC_Current))

        elif (Type_Data_Source == 2):  # $th Order Residential Thermal Model

            # We do not compute PHVAC for this case
            PHVAC = 0

        # Computing Next State
        X_Current = Building_GreyBox_Simulator(X_State, U_Measurement[:,ii], Parameter_Vector, Type_Data_Source, Type_System_Model)

        # If Else Loop: To compute state feedback based on Type of Simulation
        if (Type_Simulation == 1):  # Without State Feed Back

            # State FeedbAck from Observed Data
            X = np.hstack((X, np.reshape(X_Current, (n_State,1))))

            # Updating X_State
            X_State = X_Current
            X_State[0,0] = Y_Measurement[0,ii]

        elif (Type_Simulation == 2):  # With State Feed Back

            # State FeedbAck from Model Data
            X = np.hstack((X, np.reshape(X_Current, (n_State,1))))         

            # Updating X_State
            X_State = X_Current   

    return X, PHVAC


# =============================================================================
# Initialization
# =============================================================================

PHVAC_Sim1_Train = np.zeros((1,1))

PHVAC_Sim1_Test = np.zeros((1,1))

PHVAC_Sim2_Train = np.zeros((1,1))

PHVAC_Sim2_Test = np.zeros((1,1))


# =============================================================================
# Data Access: Dependent on Data Source
# =============================================================================

for kk in range(Total_Aggregation_Zone_Number):

    kk = kk + 1

    Aggregation_UnitNumber = kk

    print("Current Unit Number: " + str(kk))

    if (Type_Data_Source  == 1):  # PNNL Prototype Data

        # =============================================================================
        # Getting Required Data from Sim_ProcessedData
        # =============================================================================

        # Getting Current File Directory Path
        Current_FilePath = os.path.dirname(__file__)

        # Getting Folder Path
        Sim_ProcessedData_FolderPath_AggregatedTestTrain = os.path.join(Current_FilePath, '..', '..', '..', 'Results',
                                                                        'Processed_BuildingSim_Data', Simulation_Name,
                                                                        'Sim_TrainingTestingData')
        Sim_ProcessedData_FolderPath_Regression = os.path.join(Current_FilePath, '..', '..', '..', 'Results',
                                                            'Processed_BuildingSim_Data', Simulation_Name,
                                                            'Sim_RegressionModelData')

        # LOOP: Output Generation for Each Aggregated Zone

        # Creating Required File Names

        Aggregation_DF_Test_File_Name = 'Aggregation_DF_Test_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

        Aggregation_DF_Train_File_Name = 'Aggregation_DF_Train_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

        ANN_HeatInput_Test_DF_File_Name = 'ANN_HeatInput_Test_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

        ANN_HeatInput_Train_DF_File_Name = 'ANN_HeatInput_Train_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

        PHVAC_Regression_Model_File_Name = 'QAC_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk)

        # Get Required Files from Sim_AggregatedTestTrainData_FolderPath    
        AggregatedTest_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Test_File_Name), "rb")
        AggregatedTest_DF = pickle.load(AggregatedTest_Dict_File)

        AggregatedTrain_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Train_File_Name), "rb")
        AggregatedTrain_DF = pickle.load(AggregatedTrain_Dict_File)

        PHVAC_Regression_Model_File_Path = os.path.join(Sim_ProcessedData_FolderPath_Regression, PHVAC_Regression_Model_File_Name)
        PHVAC_Regression_Model = tf.keras.models.load_model(PHVAC_Regression_Model_File_Path)

        # Get Required Files from Sim_RegressionModelData_FolderPath
        ANN_HeatInput_Test_DF_File = open(os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Test_DF_File_Name),
                                            "rb")
        ANN_HeatInput_Test_DF = pickle.load(ANN_HeatInput_Test_DF_File)

        ANN_HeatInput_Train_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Train_DF_File_Name), "rb")
        ANN_HeatInput_Train_DF = pickle.load(ANN_HeatInput_Train_DF_File)

        
        # =============================================================================
        # Basic Computation
        # =============================================================================

        # Getting DateTime Data
        DateTime_Train = AggregatedTrain_DF['DateTime']
        DateTime_Test = AggregatedTest_DF['DateTime']

        # Resetting
        ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)

        # Joining Train/Test Aggregated_DF
        Aggregated_DF = pd.concat([AggregatedTrain_DF.reset_index(drop=True), AggregatedTest_DF.reset_index(drop=True)])
        Aggregated_DF.sort_values("DateTime", inplace=True)
        Aggregated_DF.reset_index(drop=True, inplace=True)

        # Joining Train/Test ANN_HeatInput_DF
        ANN_HeatInput_Train_DF.insert(0, 'DateTime', DateTime_Train)
        ANN_HeatInput_Test_DF.insert(0, 'DateTime', DateTime_Test)
        ANN_HeatInput_DF = pd.concat([ANN_HeatInput_Train_DF.reset_index(drop=True), ANN_HeatInput_Test_DF.reset_index(drop=True)])
        ANN_HeatInput_DF.sort_values("DateTime", inplace=True)
        ANN_HeatInput_DF.reset_index(drop=True, inplace=True)

        # Getting Data within Start and End Dates provided by user
        # Add DateTime to Aggregation_DF
        DateTime_List = Aggregated_DF['DateTime'].tolist() 

        # Getting Start and End Dates for the Dataset
        StartDate_Dataset = DateTime_List[0]
        EndDate_Dataset = DateTime_List[-1]

        # Getting the File Resolution from DateTime_List
        DateTime_Delta = DateTime_List[1] - DateTime_List[0]

        FileResolution_Minutes = DateTime_Delta.seconds/60

            # Getting Start and End Date
        StartDate = datetime.datetime.strptime(DateTime_Tuple[0],'%m/%d/%Y')
        EndDate = datetime.datetime.strptime(DateTime_Tuple[1],'%m/%d/%Y')

        # User Dates Corrected
        StartDate_Corrected = datetime.datetime(StartDate.year,StartDate.month,StartDate.day,0,int(FileResolution_Minutes),0)
        EndDate_Corrected = datetime.datetime(EndDate.year,EndDate.month,EndDate.day,23,60-int(FileResolution_Minutes),0)

        Counter_DateTime = -1

        # Initializing DateRange List
        DateRange_Index = []

        # FOR LOOP:
        for Element in DateTime_List:
            Counter_DateTime = Counter_DateTime + 1

            if (Element >= StartDate_Corrected and Element <= EndDate_Corrected):
                DateRange_Index.append(Counter_DateTime)

        # Getting Train and Test Dataset
        Aggregated_DF = copy.deepcopy(Aggregated_DF.iloc[DateRange_Index,:])
        ANN_HeatInput_DF = copy.deepcopy(ANN_HeatInput_DF.iloc[DateRange_Index,:])

        Aggregated_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_DF.reset_index(drop=True, inplace=True)


        # Computing QZic and QZir Train

        # Initialization
        QZic_Train = []
        QZir_Train = []
        QZic_Test = []
        QZir_Test = []
        QSol1_Test = []
        QSol1_Train = []
        QSol2_Test = []
        QSol2_Train = []
        QAC_Test = []
        QAC_Train = []

        # FOR LOOP: Getting Summation
        for ii in range(ANN_HeatInput_DF.shape[0]):
            # print(ii)
            QZic_Train_1 = ANN_HeatInput_DF['QZic_P'][ii][0] + ANN_HeatInput_DF['QZic_L'][ii][0] + \
                            ANN_HeatInput_DF['QZic_EE'][ii][0]
            QZir_Train_1 = ANN_HeatInput_DF['QZir_P'][ii][0] + ANN_HeatInput_DF['QZir_L'][ii][0] + \
                            ANN_HeatInput_DF['QZir_EE'][ii][0] + ANN_HeatInput_DF['QZivr_L'][ii][0]
            QZic_Train.append(QZic_Train_1)
            QZir_Train.append(QZir_Train_1)

            QSol1_Train_1 = ANN_HeatInput_DF['QSol1'][ii][0]
            QSol2_Train_1 = ANN_HeatInput_DF['QSol2'][ii][0]
            # QAC_Train_1 = ANN_HeatInput_DF['QAC'][ii][0]
            QAC_Train_1 = Aggregated_DF['QHVAC_X'].iloc[ii]

            QSol1_Train.append(QSol1_Train_1)
            QSol2_Train.append(QSol2_Train_1)
            QAC_Train.append(QAC_Train_1)

        ANN_HeatInput_DF.insert(2, 'QZic', QZic_Train)
        ANN_HeatInput_DF.insert(2, 'QZir', QZir_Train)
        ANN_HeatInput_DF.insert(2, 'QSol1_Corrected', QSol1_Train)
        ANN_HeatInput_DF.insert(2, 'QSol2_Corrected', QSol2_Train)
        ANN_HeatInput_DF.insert(2, 'QAC_Corrected', QAC_Train)

        ANN_HeatInput_DF.reset_index(drop=True, inplace=True)
        Aggregated_DF.reset_index(drop=True, inplace=True)

        # Creating Common DF
        Common_Data_DF = pd.concat([Aggregated_DF[[ 'DateTime', 'Zone_Air_Temperature_', 'Site_Outdoor_Air_Drybulb_Temperature_', 'System_Node_Mass_Flow_Rate_', 'System_Node_Temperature_']], ANN_HeatInput_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']]], axis=1)

        # Creating Training and Testing Split
        Common_Data_DF_Train = Common_Data_DF.iloc[0:math.floor((1-Test_Split)*len(Common_Data_DF)),:]
        
        Test_Start_Index = math.floor((1-Test_Split)*len(Common_Data_DF)) + 1

        Common_Data_DF_Test = Common_Data_DF.iloc[Test_Start_Index:,:]
        
        # Debugger
        # plt.figure()
        # plt.plot(Common_Data_DF['QAC_Corrected'])
        #plt.show()
        
        ## Creating Y_Measurement and U_Measurement - Dependent on the Model Type
        if (Type_System_Model == 1):  # One State Model

            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((6,1))
            Q_Measurement_Train = np.zeros((2,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = Common_Data_DF_Train['Zone_Air_Temperature_'].iloc[ii]

                # Getting Q Measurements
                M_Dot = Common_Data_DF_Train['System_Node_Mass_Flow_Rate_'].iloc[ii]
                Ts = Common_Data_DF_Train['System_Node_Temperature_'].iloc[ii]

                # Getting Disturbances 
                T_a = Common_Data_DF_Train['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[ii]
                Q_sol1 = Common_Data_DF_Train['QSol1_Corrected'].iloc[ii]
                Q_sol2 = Common_Data_DF_Train['QSol2_Corrected'].iloc[ii]
                Q_Zic = Common_Data_DF_Train['QZic'].iloc[ii]
                Q_Zir = Common_Data_DF_Train['QZir'].iloc[ii]

                # Getting Control
                Q_ac = Common_Data_DF_Train['QAC_Corrected'].iloc[ii]

                # Updating Y_Measurement
                Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))

                # Updating Q_Measurement
                Q_Measurement_Train = np.hstack((Q_Measurement_Train, np.reshape(np.array([M_Dot, Ts]), (2,1))))


            # Removing first elements from Y_Measurement and U_Measurement
            Y_Measurement_Train = Y_Measurement_Train[:,1:]
            U_Measurement_Train = U_Measurement_Train[:,:-1] 
            Q_Measurement_Train = Q_Measurement_Train[:,:-1] 

            # Creating Initial Condition for Testing Simulations
            X_Initial_Train = np.reshape(np.array([Common_Data_DF_Train['Zone_Air_Temperature_'].iloc[0]]*n_State), (n_State,1))


            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((6,1))
            Q_Measurement_Test = np.zeros((2,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = Common_Data_DF_Test['Zone_Air_Temperature_'].iloc[ii]

                # Getting Q Measurements
                M_Dot = Common_Data_DF_Test['System_Node_Mass_Flow_Rate_'].iloc[ii]
                Ts = Common_Data_DF_Test['System_Node_Temperature_'].iloc[ii]

                # Getting Disturbances 
                T_a = Common_Data_DF_Test['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[ii]
                Q_sol1 = Common_Data_DF_Test['QSol1_Corrected'].iloc[ii]
                Q_sol2 = Common_Data_DF_Test['QSol2_Corrected'].iloc[ii]
                Q_Zic = Common_Data_DF_Test['QZic'].iloc[ii]
                Q_Zir = Common_Data_DF_Test['QZir'].iloc[ii]

                # Getting Control
                Q_ac = Common_Data_DF_Test['QAC_Corrected'].iloc[ii]

                # Updating Y_Measurement
                Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))

                # Updating Q_Measurement
                Q_Measurement_Test = np.hstack((Q_Measurement_Test, np.reshape(np.array([M_Dot, Ts]), (2,1))))

            # Removing first elements from Y_Measurement and U_Measurement
            Y_Measurement_Test = Y_Measurement_Test[:,1:]
            U_Measurement_Test = U_Measurement_Test[:,:-1] 
            Q_Measurement_Test = Q_Measurement_Test[:,:-1] 

            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(np.array([Common_Data_DF_Test['Zone_Air_Temperature_'].iloc[0]]*n_State), (n_State,1))

        elif (Type_System_Model == 2):  # Two State Model

            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((6,1))
            Q_Measurement_Train = np.zeros((2,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = Common_Data_DF_Train['Zone_Air_Temperature_'].iloc[ii]

                # Getting Q Measurements
                M_Dot = Common_Data_DF_Train['System_Node_Mass_Flow_Rate_'].iloc[ii]
                Ts = Common_Data_DF_Train['System_Node_Temperature_'].iloc[ii]

                # Getting Disturbances 
                T_a = Common_Data_DF_Train['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[ii]
                Q_sol1 = Common_Data_DF_Train['QSol1_Corrected'].iloc[ii]
                Q_sol2 = Common_Data_DF_Train['QSol2_Corrected'].iloc[ii]
                Q_Zic = Common_Data_DF_Train['QZic'].iloc[ii]
                Q_Zir = Common_Data_DF_Train['QZir'].iloc[ii]

                # Getting Control
                Q_ac = Common_Data_DF_Train['QAC_Corrected'].iloc[ii]

                # Updating Y_Measurement
                Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))

                # Updating Q_Measurement
                Q_Measurement_Train = np.hstack((Q_Measurement_Train, np.reshape(np.array([M_Dot, Ts]), (2,1))))

            # Removing first elements from Y_Measurement and U_Measurement
            Y_Measurement_Train = Y_Measurement_Train[:,1:]
            U_Measurement_Train = U_Measurement_Train[:,:-1] 
            Q_Measurement_Train = Q_Measurement_Train[:,:-1] 

            # Creating Initial Condition for Testing Simulations
            X_Initial_Train = np.reshape(np.array([Common_Data_DF_Train['Zone_Air_Temperature_'].iloc[0]]*n_State), (n_State,1))

            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((6,1))
            Q_Measurement_Test = np.zeros((2,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = Common_Data_DF_Test['Zone_Air_Temperature_'].iloc[ii]

                # Getting Q Measurements
                M_Dot = Common_Data_DF_Test['System_Node_Mass_Flow_Rate_'].iloc[ii]
                Ts = Common_Data_DF_Test['System_Node_Temperature_'].iloc[ii]

                # Getting Disturbances 
                T_a = Common_Data_DF_Test['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[ii]
                Q_sol1 = Common_Data_DF_Test['QSol1_Corrected'].iloc[ii]
                Q_sol2 = Common_Data_DF_Test['QSol2_Corrected'].iloc[ii]
                Q_Zic = Common_Data_DF_Test['QZic'].iloc[ii]
                Q_Zir = Common_Data_DF_Test['QZir'].iloc[ii]

                # Getting Control
                Q_ac = Common_Data_DF_Test['QAC_Corrected'].iloc[ii]

                # Updating Y_Measurement
                Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))

                # Updating Q_Measurement
                Q_Measurement_Test = np.hstack((Q_Measurement_Test, np.reshape(np.array([M_Dot, Ts]), (2,1))))

            # Removing first elements from Y_Measurement and U_Measurement
            Y_Measurement_Test = Y_Measurement_Test[:,1:]
            U_Measurement_Test = U_Measurement_Test[:,:-1] 
            Q_Measurement_Test = Q_Measurement_Test[:,:-1] 

            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(np.array([Common_Data_DF_Test['Zone_Air_Temperature_'].iloc[0]]*n_State), (n_State,1))   
    

    elif (Type_Data_Source  == 2):  # Simulated 4 State House Thermal Model Data

        # =============================================================================
        # Getting Required Data from Sim_ProcessedData
        # =============================================================================

        # Getting Current File Directory Path
        Current_FilePath = os.path.dirname(__file__)

        # Getting Folder Path
        Sim_HouseThermalData_FilePath = os.path.join(Current_FilePath, '..', '..', '..', 'Data',
                                                                        'HouseThermalModel_Data', 'PostProcessedFiles',
                                                                        'LargeHouse', Simulated_HouseThermalData_Filename)
        


        ## Read Dictionary from .mat File
        Common_Data_Dict = scipy.io.loadmat(Sim_HouseThermalData_FilePath)

        ## Reading ActuaL Data
        Common_Data_Array = Common_Data_Dict[Simulated_HouseThermalData_Filename.split('.')[0]]

        # =============================================================================
        # Basic Computation
        # =============================================================================

        # Getting Start and End Date
        StartDate = datetime.datetime.strptime(DateTime_Tuple[0],'%m/%d/%Y')
        EndDate = datetime.datetime.strptime(DateTime_Tuple[1],'%m/%d/%Y')

        # User Dates Corrected
        StartDate_Corrected = datetime.datetime(StartDate.year,StartDate.month,StartDate.day,0,int(FileResolution_Minutes),0)
        EndDate_Corrected = datetime.datetime(EndDate.year,EndDate.month,EndDate.day,23,60-int(FileResolution_Minutes),0)

        Counter_DateTime = -1

        # Initializing DateRange List
        DateRange_Index = []

        # FOR LOOP:
        for Element in DateTime_List:
            Counter_DateTime = Counter_DateTime + 1

            if (Element >= StartDate_Corrected and Element <= EndDate_Corrected):
                DateRange_Index.append(Counter_DateTime)

        # Getting Train and Test Dataset
        Common_Data_Array_DateCorrected = Common_Data_Array[DateRange_Index,:]

        # Creating Training and Testing Split
        Common_Data_Array_DateCorrected_Train = Common_Data_Array_DateCorrected[0:math.floor((1-Test_Split)*Common_Data_Array_DateCorrected.shape[0]),:]
        
        Test_Start_Index = math.floor((1-Test_Split)*Common_Data_Array_DateCorrected.shape[0]) + 1

        Common_Data_Array_DateCorrected_Test = Common_Data_Array_DateCorrected[Test_Start_Index:,:]

        ## Reading DateTime/States(k+1)/States(k)/Input(k) - Training
        DateTime_Array_Train = Common_Data_Array_DateCorrected_Train[:,0:4]
        States_k_1_Array_Train = Common_Data_Array_DateCorrected_Train[:,4:8]
        States_k_Array_Train = Common_Data_Array_DateCorrected_Train[:,8:12]
        Inputs_k_Array_Train = Common_Data_Array_DateCorrected_Train[:,12:16]

        ## Reading DateTime/States(k+1)/States(k)/Input(k) - Testing
        DateTime_Array_Test = Common_Data_Array_DateCorrected_Test[:,0:4]
        States_k_1_Array_Test = Common_Data_Array_DateCorrected_Test[:,4:8]
        States_k_Array_Test = Common_Data_Array_DateCorrected_Test[:,8:12]
        Inputs_k_Array_Test = Common_Data_Array_DateCorrected_Test[:,12:16]        

        ## Creating Y_Measurement and U_Measurement - Dependent on the Model Type
        if (Type_System_Model == 1):  # One State Model

            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((6,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = States_k_1_Array_Train[ii,1]

                # Getting Disturbances 
                T_a = Inputs_k_Array_Train[ii,0]
                Q_int = Inputs_k_Array_Train[ii,3]
                Q_ac = Inputs_k_Array_Train[ii,4]
                Q_venti = Inputs_k_Array_Train[ii,5]
                Q_infil = Inputs_k_Array_Train[ii,6]
                Q_sol = Inputs_k_Array_Train[ii,7]

                # Updating Y_Measurement
                Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (6,1))))

            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Train = Y_Measurement_Train[:,1:]
            # U_Measurement_Train = U_Measurement_Train[:,:-1] 
                
            # Creating Initial Condition for Testing Simulations
            X_Initial_Train = np.reshape(States_k_Array_Train[0,0:n_State], (n_State,1))   

            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((6,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = States_k_1_Array_Test[ii,1]

                # Getting Disturbances 
                T_a = Inputs_k_Array_Test[ii,0]
                Q_int = Inputs_k_Array_Test[ii,3]
                Q_ac = Inputs_k_Array_Test[ii,4]
                Q_venti = Inputs_k_Array_Test[ii,5]
                Q_infil = Inputs_k_Array_Test[ii,6]
                Q_sol = Inputs_k_Array_Test[ii,7]

                # Updating Y_Measurement
                Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (6,1))))

            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Test = Y_Measurement_Test[:,1:]
            # U_Measurement_Test = U_Measurement_Test[:,:-1]
                
            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(States_k_Array_Test[0,0:n_State], (n_State,1))  

            # Q_Mesurement Place Holders
            Q_Measurement_Train = np.zeros((2,1))
            Q_Measurement_Test = np.zeros((2,1))

        elif (Type_System_Model == 2):  # Two State Model

            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((7,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = States_k_1_Array_Train[ii,1]

                # Getting Disturbances 
                T_sol_w = Inputs_k_Array_Train[ii,1]
                T_a = Inputs_k_Array_Train[ii,0]
                Q_int = Inputs_k_Array_Train[ii,3]
                Q_ac = Inputs_k_Array_Train[ii,4]
                Q_venti = Inputs_k_Array_Train[ii,5]
                Q_infil = Inputs_k_Array_Train[ii,6]
                Q_sol = Inputs_k_Array_Train[ii,7]

                # Updating Y_Measurement
                Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_sol_w, T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (7,1))))

            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Train = Y_Measurement_Train[:,1:]
            # U_Measurement_Train = U_Measurement_Train[:,:-1] 
                
            # Creating Initial Condition for Testing Simulations
            X_Initial_Train = np.reshape(States_k_Array_Train[0,0:n_State], (n_State,1))  

            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((7,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = States_k_1_Array_Test[ii,1]

                # Getting Disturbances 
                T_sol_w = Inputs_k_Array_Test[ii,1]
                T_a = Inputs_k_Array_Test[ii,0]
                Q_int = Inputs_k_Array_Test[ii,3]
                Q_ac = Inputs_k_Array_Test[ii,4]
                Q_venti = Inputs_k_Array_Test[ii,5]
                Q_infil = Inputs_k_Array_Test[ii,6]
                Q_sol = Inputs_k_Array_Test[ii,7]

                # Updating Y_Measurement
                Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_sol_w, T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (7,1))))

            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Test = Y_Measurement_Test[:,1:]
            # U_Measurement_Test = U_Measurement_Test[:,:-1]
                
            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(States_k_Array_Test[0,0:n_State], (n_State,1))  

            # Q_Mesurement Place Holders
            Q_Measurement_Train = np.zeros((2,1))
            Q_Measurement_Test = np.zeros((2,1))


        elif (Type_System_Model == 4):  # Four State Model #Pandas data frame with column name that was read from .mat file

            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((8,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = States_k_1_Array_Train[ii,1]

                # Getting Disturbances 
                T_sol_w = Inputs_k_Array_Train[ii,1]
                T_sol_r = Inputs_k_Array_Train[ii,2]
                T_a = Inputs_k_Array_Train[ii,0]
                Q_int = Inputs_k_Array_Train[ii,3]
                Q_ac = Inputs_k_Array_Train[ii,4]
                Q_venti = Inputs_k_Array_Train[ii,5]
                Q_infil = Inputs_k_Array_Train[ii,6]
                Q_sol = Inputs_k_Array_Train[ii,7]

                # Updating Y_Measurement
                Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_sol_w, T_sol_r, T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (8,1))))

            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Train = Y_Measurement_Train[:,1:]
            # U_Measurement_Train = U_Measurement_Train[:,:-1] 
                
            # Creating Initial Condition for Testing Simulations
            X_Initial_Train = np.reshape(States_k_Array_Train[0,0:n_State], (n_State,1))  

            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((7,1))

            for ii in range(Aggregated_DF.shape[0]):

                # Getting State Measurement
                T_z = States_k_1_Array_Test[ii,1]

                # Getting Disturbances 
                T_sol_w = Inputs_k_Array_Test[ii,1]
                T_sol_r = Inputs_k_Array_Test[ii,2]
                T_a = Inputs_k_Array_Test[ii,0]
                Q_int = Inputs_k_Array_Test[ii,3]
                Q_ac = Inputs_k_Array_Test[ii,4]
                Q_venti = Inputs_k_Array_Test[ii,5]
                Q_infil = Inputs_k_Array_Test[ii,6]
                Q_sol = Inputs_k_Array_Test[ii,7]

                # Updating Y_Measurement
                Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))

                # Updating U_Measurement
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_sol_w, T_sol_r, T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (8,1))))

            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Test = Y_Measurement_Test[:,1:]
            # U_Measurement_Test = U_Measurement_Test[:,:-1]   
                
            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(States_k_Array_Test[0,0:n_State], (n_State,1))  

            # Q_Mesurement Place Holders
            Q_Measurement_Train = np.zeros((2,1))
            Q_Measurement_Test = np.zeros((2,1))


    # =============================================================================
    # Creating Sim_ANNModelData Folder
    # =============================================================================

    if (Shoter_ResultsPath == False):

        # Making Additional Folders for storing Aggregated Files
        Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..', '..', 'Results',
                                                            'Processed_BuildingSim_Data')
            
        Sim_ANNModelData_FolderName = 'Sim_GB_ModelData'      
    

        # Checking if Folders Exist if not create Folders
        if (
                os.path.isdir(
                    os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName))):

            # Folders Exist
            z = None

        else:

            os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName))

        # Make the Training Folder
        os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName, Training_FolderName))

        # Creating Sim_RegressionModelData Folder Path
        Sim_ANNModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name,
                                                Sim_ANNModelData_FolderName, Training_FolderName) 
    
    elif (Shoter_ResultsPath == True):
    
        ## Shorter Path
        
        Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..', '..')

        Sim_ANNModelData_FolderName = Short_ResultFolder

        # Make the Training Folder
        os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName, Training_FolderName))

        # Creating Sim_RegressionModelData Folder Path
        Sim_ANNModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath,
                                                Sim_ANNModelData_FolderName, Training_FolderName)


    
    # =============================================================================
    # Filter Creation
    # =============================================================================
        
    # Timing Method
    Sim_StartTime = time.time()

    # Creating initial Filter mean vector
    if (Type_Data_Source  == 1):  # PNNL Prototype Data

        if (Type_System_Model == 1):  # One State Model

            m_ini_model = np.reshape(np.array([Y_Measurement_Train[:,0][0], Theta_Initial[0], Theta_Initial[1], Theta_Initial[2],]), (n,1))

        elif (Type_System_Model == 2):  # Two State Model

            m_ini_model = np.reshape(np.array([Y_Measurement_Train[:,0][0], Y_Measurement_Train[:,0][0], Theta_Initial[0], Theta_Initial[1], Theta_Initial[2], Theta_Initial[3], Theta_Initial[4], Theta_Initial[5],]), (n,1))

    elif (Type_Data_Source  == 2):  # Simulated 4 State House Thermal Model Data

        if (Type_System_Model == 1):  # One State Model

            m_ini_model = np.reshape(np.array([Y_Measurement_Train[:,0][0], Theta_Initial[0], Theta_Initial[1], Theta_Initial[2], Theta_Initial[3], Theta_Initial[4],]), (n,1))

        elif (Type_System_Model == 2):  # Two State Model

            m_ini_model = np.reshape(np.array([Y_Measurement_Train[:,0][0], Y_Measurement_Train[:,0][0], Theta_Initial[0], Theta_Initial[1], Theta_Initial[2], Theta_Initial[3], Theta_Initial[4], Theta_Initial[5], Theta_Initial[6], Theta_Initial[7], Theta_Initial[8],]), (n,1))

        elif (Type_System_Model == 4):  # Four State Model

            m_ini_model = np.reshape(np.array([Y_Measurement_Train[:,0][0], Y_Measurement_Train[:,0][0], Y_Measurement_Train[:,0][0], Y_Measurement_Train[:,0][0], Theta_Initial[0], Theta_Initial[1], Theta_Initial[2], Theta_Initial[3], Theta_Initial[4], Theta_Initial[5], Theta_Initial[6], Theta_Initial[7], Theta_Initial[8], Theta_Initial[9], Theta_Initial[10], Theta_Initial[11],]), (n,1))

    # Creating initial Filter state covariance matrix
    P_ini_model = P_model*np.eye(n)

    # Create the model Q and R matrices
    Q_d = np.eye(n) * np.reshape(np.array([Q_model, Q_model, Q_params,Q_params,Q_params,Q_params,Q_params,Q_params]), (n,1))
    R_d = np.reshape(np.array([R_model]), (1,1))

    # If Elseif Loop: For type of Filter chosen
    if (StateAug_Filter == 1):

        SingleZone_Filter = EKFS(SingleZone_f1, SingleZone_F, SingleZone_h1, SingleZone_H, m_ini_model, P_ini_model, Q_d, R_d)

    elif (StateAug_Filter == 2):

        SingleZone_Filter = UKFS(SingleZone_f1, SingleZone_h1, n, m, alpha, k, beta,  m_ini_model, P_ini_model, Q_d, R_d)

    elif (StateAug_Filter == 3):

        SingleZone_Filter = GFS(1, SingleZone_f1, SingleZone_h1, n, m,  m_ini_model, P_ini_model, Q_d, R_d, p)

    elif (StateAug_Filter == 4):

        SingleZone_Filter = GFS(2, SingleZone_f1, SingleZone_h1, n, m,  m_ini_model, P_ini_model, Q_d, R_d, p)


    # =============================================================================
    # Filter Time Evolution
    # =============================================================================

    # Initializing model filter state array to store time evolution
    x_model_nonlinear_filter = m_ini_model

    # If Elseif Loop: For type of Filter chosen
    if (StateAug_Filter == 1):

        # FOR LOOP: For each discrete time-step
        for ii in range(Y_Measurement_Train.shape[1]-1):
            
            ## For measurements coming from Nonlinear System
            
            # Extended Kalman Filter: Predict Step    
            m_k_, P_k_ = SingleZone_Filter.Extended_Kalman_Predict(np.reshape(U_Measurement_Train[:,ii],(U_Measurement_Train.shape[0],1)))
            
            # Extended Kalman Filter: Update Step
            v_k, S_k, K_k = SingleZone_Filter.Extended_Kalman_Update(np.reshape(Y_Measurement_Train[:,ii+1][0], (1,1)), m_k_, P_k_)    
            
            # Storing the Filtered states
            x_k_filter = SingleZone_Filter.m_k
            x_model_nonlinear_filter = np.hstack((x_model_nonlinear_filter, x_k_filter))

    elif (StateAug_Filter == 2):

        # FOR LOOP: For each discrete time-step
        for ii in range(Y_Measurement_Train.shape[1]-1):
        
            ## For measurements coming from Nonlinear System
            
            # Extended Kalman Filter: Predict Step    
            m_k_, P_k_, D_k = SingleZone_Filter.Unscented_Kalman_Predict(np.reshape(U_Measurement_Train[:,ii],(U_Measurement_Train.shape[0],1)))
            
            # Extended Kalman Filter: Update Step
            mu_k, S_k, C_k, K_k = SingleZone_Filter.Unscented_Kalman_Update(np.reshape(Y_Measurement_Train[:,ii+1][0], (1,1)), m_k_, P_k_)    
            
            # Storing the Filtered states
            x_k_filter = SingleZone_Filter.m_k
            x_model_nonlinear_filter = np.hstack((x_model_nonlinear_filter, x_k_filter))

    elif ((StateAug_Filter == 3) or (StateAug_Filter == 4)):

        # FOR LOOP: For each discrete time-step
        for ii in range(Y_Measurement_Train.shape[1]-1):
        
            ## For measurements coming from Nonlinear System
            
            # Extended Kalman Filter: Predict Step    
            m_k_, P_k_, D_k = SingleZone_Filter.Gaussian_Predict(np.reshape(U_Measurement_Train[:,ii],(U_Measurement_Train.shape[0],1)))
            
            # Extended Kalman Filter: Update Step
            mu_k, S_k, C_k, K_k = SingleZone_Filter.Gaussian_Update(np.reshape(Y_Measurement_Train[:,ii+1][0], (1,1)), m_k_, P_k_)    
            
            # Storing the Filtered states
            x_k_filter = SingleZone_Filter.m_k
            x_model_nonlinear_filter = np.hstack((x_model_nonlinear_filter, x_k_filter))    

    # =============================================================================
    # Smoother Time Evolution
    # =============================================================================

    ## Create the input vector list
    U_Measurement_list = [] 

    for ii in range(U_Measurement_Train.shape[1]):
        
        U_Measurement_list.append(np.reshape(U_Measurement_Train[:,ii], (U_Measurement_Train.shape[0],1)))
        
    ## Create the measurement vector list
    Y_Measurement_list = []

    for ii in range(Y_Measurement_Train.shape[1]):
        
        Y_Measurement_list.append(np.reshape(Y_Measurement_Train[:,ii], (1,1)))



    # If Elseif Loop: For type of Filter chosen
    if (StateAug_Filter == 1):

        # Running Smoother
        G_k_list, m_k_s_list, P_k_s_list = SingleZone_Filter.Extended_Kalman_Smoother(U_Measurement_list, Y_Measurement_list[1:])     
            

    elif (StateAug_Filter == 2):

        # Running Smoother
        G_k_list, m_k_s_list, P_k_s_list = SingleZone_Filter.Unscented_Kalman_Smoother(U_Measurement_list, Y_Measurement_list[1:]) 
            

    elif ((StateAug_Filter == 3) or (StateAug_Filter == 4)):

        # Running Smoother
        G_k_list, m_k_s_list, P_k_s_list = SingleZone_Filter.Gaussian_Smoother(U_Measurement_list, Y_Measurement_list[1:]) 

    # Storing the Filtered states
    for ii in range(len(m_k_s_list)):
        
        if (ii == 0):
            
            x_model_nonlinear_smoother = m_k_s_list[ii]
            
        else:
            
            x_model_nonlinear_smoother = np.hstack((x_model_nonlinear_smoother, m_k_s_list[ii]))


    # Method Simulation
    Sim_EndTime = time.time()

    MethodTime = (Sim_EndTime - Sim_StartTime)/(Y_Measurement_Train.shape[1])

    # =============================================================================
    # Getting Building RC-Network Model Parameters - Parameter_Vector
    # =============================================================================
    # IF Else Loop: Based on Type_Data_Source
    if (Type_Data_Source == 1):  # PNNL Prototype Data

        # IF Else Loop: Based on Type_System_Model
        if (Type_System_Model == 1):  # 1-State RC-Network

            Parameter_Vector = np.reshape(x_model_nonlinear_smoother[-1][1:,0],(n_Parameter,1))

        elif (Type_System_Model == 2):  # 2-State RC-Network

            Parameter_Vector = np.reshape(x_model_nonlinear_smoother[-1][2:,0],(n_Parameter,1))

    elif (Type_Data_Source == 2):  # 4th Order House Thermal Model

        # IF Else Loop: Based on Type_System_Model
        if (Type_System_Model == 1):  # 1-State RC-Network

            Parameter_Vector = np.reshape(x_model_nonlinear_smoother[-1][1:,0],(n_Parameter,1))

        elif (Type_System_Model == 2):  # 2-State RC-Network

            Parameter_Vector = np.reshape(x_model_nonlinear_smoother[-1][2:,0],(n_Parameter,1))

        elif (Type_System_Model == 4):  # 4-State RC-Network

            Parameter_Vector = np.reshape(x_model_nonlinear_smoother[-1][4:,0],(n_Parameter,1))

    # =============================================================================
    # Simulation1 Testing : on Training Data (Feedback from observed state data)
    # =============================================================================   

    # Timing Simulation
    Sim_StartTime = time.time()

    X_Sim1_Train, PHVAC_Sim1_Train_Current = Building_GreyBox_Simulator_Data_Sim(Y_Measurement_Train, U_Measurement_Train, Q_Measurement_Train, X_Initial_Train, Parameter_Vector, Type_Data_Source, Type_System_Model, 1)

    # Timing Simulation
    Sim_EndTime = time.time()

    SimTime = (Sim_EndTime - Sim_StartTime)/(Y_Measurement_Train.shape[1])

    # =============================================================================
    # Simulation1 Testing : on Testing Data (Feedback from observed state data)
    # =============================================================================
    
    X_Sim1_Test, PHVAC_Sim1_Test_Current  = Building_GreyBox_Simulator_Data_Sim(Y_Measurement_Test, U_Measurement_Test, Q_Measurement_Test, X_Initial_Test, Parameter_Vector, Type_Data_Source, Type_System_Model, 1)

    # =============================================================================
    # Simulation2 Testing : on Training Data (Feedback from states generated by model)
    # =============================================================================

    X_Sim2_Train, PHVAC_Sim2_Train_Current  = Building_GreyBox_Simulator_Data_Sim(Y_Measurement_Train, U_Measurement_Train, Q_Measurement_Train, X_Initial_Train, Parameter_Vector, Type_Data_Source, Type_System_Model, 2)     

    # =============================================================================
    # Simulation2 Testing : on Testing Data (Feedback from states generated by model)
    # =============================================================================

    X_Sim2_Test, PHVAC_Sim2_Test_Current  = Building_GreyBox_Simulator_Data_Sim(Y_Measurement_Test, U_Measurement_Test, Q_Measurement_Test, X_Initial_Test, Parameter_Vector, Type_Data_Source, Type_System_Model, 2)

    


    # =============================================================================
    # Computing Actual PHVAC
    # =============================================================================  
    # IF Else Loop: Based on Type_Data_Source
    if (Type_Data_Source == 1):  # PNNL Prototype Data

        PHVAC_Sim1_Train = PHVAC_Sim1_Train + PHVAC_Sim1_Train_Current

        PHVAC_Sim1_Test = PHVAC_Sim1_Test + PHVAC_Sim1_Test_Current

        PHVAC_Sim2_Train = PHVAC_Sim2_Train + PHVAC_Sim2_Train_Current

        PHVAC_Sim2_Test = PHVAC_Sim2_Train + PHVAC_Sim2_Test_Current

    # =============================================================================
    # Computing : Accuracy/Error/Training Time/Simulation Time on Training and Testing Data 
    # =============================================================================

    # Sim1 Train Error/Accuracy   
    Y_Train_Sim1_Predicted = np.reshape(X_Sim1_Train[0,1:],(Y_Measurement_Train.shape[1],1))
    Y_Train_Sim1_Actual = np.reshape(Y_Measurement_Train,(Y_Measurement_Train.shape[1],1))

    Train_Sim1_PercentageError = np.mean((np.absolute((Y_Train_Sim1_Predicted-Y_Train_Sim1_Actual)/(Y_Train_Sim1_Actual))) *100)

    Train_Sim1_PercentageAccuracy = 1 - Train_Sim1_PercentageError

    # Sim1 Test Accuracy Error/Accuracy          
    Y_Test_Sim1_Predicted = np.reshape(X_Sim1_Test[0,1:],(Y_Measurement_Test.shape[1],1))
    Y_Test_Sim1_Actual = np.reshape(Y_Measurement_Test,(Y_Measurement_Test.shape[1],1))

    Test_Sim1_PercentageError = np.mean((np.absolute((Y_Test_Sim1_Predicted-Y_Test_Sim1_Actual)/(Y_Test_Sim1_Actual))) *100)

    Test_Sim1_PercentageAccuracy = 1 - Test_Sim1_PercentageError

    # Sim2 Train Accuracy Error/Accuracy         
    Y_Train_Sim2_Predicted = np.reshape(X_Sim2_Train[0,1:],(Y_Measurement_Train.shape[1],1))
    Y_Train_Sim2_Actual = np.reshape(Y_Measurement_Train,(Y_Measurement_Train.shape[1],1))

    Train_Sim2_PercentageError = np.mean((np.absolute((Y_Train_Sim2_Predicted-Y_Train_Sim2_Actual)/(Y_Train_Sim2_Actual))) *100)

    Train_Sim2_PercentageAccuracy = 1 - Train_Sim2_PercentageError

    # Sim2 Test Accuracy Error/Accuracy 
    Y_Test_Sim2_Predicted = np.reshape(X_Sim2_Test[0,1:],(Y_Measurement_Test.shape[1],1))
    Y_Test_Sim2_Actual = np.reshape(Y_Measurement_Test,(Y_Measurement_Test.shape[1],1))

    Test_Sim2_PercentageError = np.mean((np.absolute((Y_Test_Sim2_Predicted-Y_Test_Sim2_Actual)/(Y_Test_Sim2_Actual))) *100)

    Test_Sim2_PercentageAccuracy = 1 - Test_Sim2_PercentageError

    # Appending Percentage Accuracy into Table
    GB_PercentageAccuracy_Current_DF = pd.DataFrame([[GBModel_Key, Aggregation_UnitNumber, Train_Sim1_PercentageError, Test_Sim1_PercentageError, Train_Sim2_PercentageError, Test_Sim2_PercentageError, Train_Sim1_PercentageAccuracy, Test_Sim1_PercentageAccuracy, Train_Sim2_PercentageAccuracy, Test_Sim2_PercentageAccuracy, MethodTime, SimTime]],columns=['GB Model Name', 'Agg Zone Num', 'Sim1 Train Mean Error', 'Sim1 Test Mean Error', 'Sim2 Train Mean Error', 'Sim2 Test Mean Error', 'Sim1 Train Mean Acc', 'Sim1 Test Mean Acc', 'Sim2 Train Mean Acc', 'Sim2 Test Mean Acc','Method Time/Iteration', 'Sim Time/Iteration'])
    GB_PercentageAccuracy_DF = pd.concat([GB_PercentageAccuracy_DF, GB_PercentageAccuracy_Current_DF], ignore_index=True)

    # Appending Parameters into Table
    Parameter_Data_List = [GBModel_Key] + [Aggregation_UnitNumber] + Parameter_Vector.tolist() + Theta_Initial
    Parameter_ColumnName_List = ['GB Model Name'] + ['Agg Zone Num'] + Theta_Names + [x+'_ini' for x in Theta_Names]

    GB_Parameters_Current_DF = pd.DataFrame([Parameter_Data_List],columns=Parameter_ColumnName_List)
    GB_PercentageAccuracy_DF = pd.concat([GB_PercentageAccuracy_DF, GB_PercentageAccuracy_Current_DF], ignore_index=True)

    # =============================================================================
    # Storing Result Time Series in Pickle Files for better visualization later (Add DateTime)
    # =============================================================================

    # =============================================================================
    # Plotting : Filter
    # =============================================================================
    # Setting Figure Size
    plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

    # Plotting Figures
    plt.figure()

    # Plotting True vs. Filtered States of Nonlinear System
    plt.subplot(411)
    plt.plot(Y_Measurement_Train[:,:].transpose(), linestyle='-', linewidth=3, label=r'$T_{z_{meas}}$ $(^{ \circ }C)$')
    plt.plot(U_Measurement_Train[0,:].transpose(), linestyle='-', linewidth=3, label=r'$T_{a_{meas}}$ $(^{ \circ }C)$')
    plt.plot(x_model_nonlinear_filter[0:2,:].transpose(), linestyle='--', linewidth=1, label=[r'$T_{z_{filter}}$ $(^{ \circ }C)$', r'$T_{w_{filter}}$ $(^{ \circ }C)$'])
    plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
    plt.ylabel('States '+ r'$(x)$', fontsize=12)
    plt.title('Single Zone - Measured vs. Filtered States' + GBModel_Key, fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)

    # Plotting True vs. Filtered Parameters of Nonlinear System  $(^{ \circ }C / W)$  $(J / ^{ \circ }C)$
    plt.subplot(412)
    plt.plot(x_model_nonlinear_filter[2:4,:].transpose(), linestyle='--', linewidth=1, label=[r'$R_{zw_{filter}}$ $(^{ \circ }C / W)$', r'$R_{wa_{filter}}$ $(^{ \circ }C / W)$'])
    plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
    plt.ylabel('Parameter '+ r'$(R_{zw},R_{wa})$', fontsize=12)
    plt.title('Single Zone - Filtered Parameters ' + r'$R_{zw},R_{wa}$ ' + GBModel_Key, fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(413)
    plt.plot(x_model_nonlinear_filter[4:6,:].transpose(), linestyle='--', linewidth=1, label=[r'$C_{z_{filter}}$ $(J / ^{ \circ }C)$', r'$C_{w_{filter}}$ $(J / ^{ \circ }C)$'])
    plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
    plt.ylabel('Parameter '+ r'$(C_{z}, C_{w})$', fontsize=12)
    plt.title('Single Zone - Filtered Parameters ' + r'$C_{z}, C_{w}$ ' + GBModel_Key, fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(414)
    plt.plot(x_model_nonlinear_filter[6:8,:].transpose(), linestyle='--', linewidth=1, label=[r'$A_{sol1_{filter}}$ ', r'$A_{sol2_{filter}}$'])
    plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
    plt.ylabel('Parameter '+ r'$(A_{sol1}, A_{sol2})$', fontsize=12)
    plt.title('Single Zone - Filtered Parameters  ' + r'$A_{sol1}, A_{sol2}$ ' + GBModel_Key, fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)


    #plt.show()
    # =============================================================================
    # Plotting : Filter
    # =============================================================================
    # Setting Figure Size
    plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

    # Plotting Figures
    plt.figure()

    # Plotting True vs. Smoothed States of Nonlinear System
    plt.subplot(411)
    plt.plot(Y_Measurement[:,:].transpose(), linestyle='-', linewidth=3, label=r'$T_{z_{meas}}$ $(^{ \circ }C)$')
    plt.plot(U_Measurement[0,:].transpose(), linestyle='-', linewidth=3, label=r'$T_{a_{meas}}$ $(^{ \circ }C)$')
    plt.plot(x_model_nonlinear_smoother[0:2,:].transpose(), linestyle='--', linewidth=1, label=[r'$T_{z_{smooth}}$ $(^{ \circ }C)$', r'$T_{w_{smooth}}$ $(^{ \circ }C)$'])
    plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
    plt.ylabel('States '+ r'$(x)$', fontsize=12)
    plt.title('Single Zone - Measured vs. Smoothed States' + GBModel_Key, fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)

    # Plotting True vs. Smoothed Parameters of Nonlinear System  $(^{ \circ }C / W)$  $(J / ^{ \circ }C)$
    plt.subplot(412)
    plt.plot(x_model_nonlinear_smoother[2:4,:].transpose(), linestyle='--', linewidth=1, label=[r'$R_{zw_{smooth}}$ $(^{ \circ }C / W)$', r'$R_{wa_{smooth}}$ $(^{ \circ }C / W)$'])
    plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
    plt.ylabel('Parameter '+ r'$(R_{zw},R_{wa})$', fontsize=12)
    plt.title('Single Zone - Smoothed Parameters ' + r'$R_{zw},R_{wa}$ ' + GBModel_Key, fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(413)
    plt.plot(x_model_nonlinear_smoother[4:6,:].transpose(), linestyle='--', linewidth=1, label=[r'$C_{z_{smooth}}$ $(J / ^{ \circ }C)$', r'$C_{w_{smooth}}$ $(J / ^{ \circ }C)$'])
    plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
    plt.ylabel('Parameter '+ r'$(C_{z}, C_{w})$', fontsize=12)
    plt.title('Single Zone - Smoothed Parameters ' + r'$C_{z}, C_{w}$ ' + GBModel_Key, fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(414)
    plt.plot(x_model_nonlinear_smoother[6:8,:].transpose(), linestyle='--', linewidth=1, label=[r'$A_{sol1_{smooth}}$ ', r'$A_{sol2_{smooth}}$'])
    plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
    plt.ylabel('Parameter '+ r'$(A_{sol1}, A_{sol2})$', fontsize=12)
    plt.title('Single Zone - Smoothed Parameters  ' + r'$A_{sol1}, A_{sol2}$ ' + GBModel_Key, fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)


    plt.show()