#-------------------------------------------------------------------Initial Setup------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#

## Importing external modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

## Setting up path for custom modules

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

# Attributes of Extended_Kalman_Filter_Smoother class
dir(EKFS)
dir(UKFS)
dir(GFS)

#--------------------------------------------------------------Defining True System----------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
# Defining the Simple Pendulum Discrete-Time Nonlinear System Function
def SimplePendulum_f(x_k_1, u_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        u_k_1 (numpy.array): Previous system measurement
        
    Returns:
        x_k_true (numpy.array): Current true system state
        x_k (numpy.array): Current noise corrupted system state       
    """
    
    # Simulation parameters
    g = 9.86
    L = 1.
    Q = 0.01
    R = 0.001
    ts = 0.001
    
    # Create the true Q and R matrices
    Q = np.reshape(np.array([[Q, 0],[0, Q]]), (2,2))
    R = np.reshape(np.array([R]), (1,1))
    
    # Getting individual state components
    theta_k_1 = x_k_1[0,0]
    omega_k_1 = x_k_1[1,0]
    
    # Computing current true system state
    theta_k = theta_k_1 + ts*(omega_k_1)
    omega_k = omega_k_1 + ts*(-(g/L)*np.sin(theta_k_1)) 
                                        
    x_k_true = np.reshape(np.array([theta_k, omega_k]),(2,1))
    
    # Computing Process/Measurement Noise
    q_k_1 = np.random.multivariate_normal(np.array([0,0]), Q)
    q_k_1 = np.reshape(q_k_1, (2,1))
                                        
    # Computing noise corrupted system state/measurement
    x_k = x_k_true + q_k_1
                                        
    # Return statement
    return x_k_true, x_k 

# Defining the Simple Pendulum Discrete-Time Nonlinear Measurement Function
def SimplePendulum_h(x_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        
    Returns:
        y_k_true (numpy.array): Current true system measurement
        y_k (numpy.array): Current noise corrupted system measurement        
    """
    
    # Simulation parameters
    g = 9.86
    L = 1.
    Q = 0.01
    R = 0.001
    
    # Create the true Q and R matrices
    Q = np.reshape(np.array([[Q, 0],[0, Q]]), (2,2))
    R = np.reshape(np.array([R]), (1,1))
    
    # Getting individual state components
    theta_k_1 = x_k_1[0,0]
    omega_k_1 = x_k_1[1,0]
    
    # Computing current true system state
    omega_k = omega_k_1  
    
    # Computing current true system measurement
    y_k_true = omega_k
    y_k_true = np.reshape(y_k_true,(1,1))
    
    # Computing Process/Measurement Noise                                        
    r_k = np.random.multivariate_normal(np.array([0]), R)
                                        
    # Computing noise corrupted system state/measurement
    y_k = y_k_true + r_k
                                        
    # Return statement
    return y_k_true, y_k 


#--------------------------------------------------------------Defining System Model---------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
# Defining the Simple Pendulum Discrete-Time Nonlinear System Function
def SimplePendulum_f1(x_k_1, u_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        u_k_1 (numpy.array): Previous system measurement
        
    Returns:
        x_k_true (numpy.array): Current true system state
        x_k (numpy.array): Current noise corrupted system state       
    """
    
    # Simulation parameters
    ts = 0.001    
       
    # Getting individual state components
    theta_k_1 = x_k_1[0,0]
    omega_k_1 = x_k_1[1,0]

    # Getting the parameter components
    g_k_1 = x_k_1[2,0]
    L_k_1 = x_k_1[3,0]
    
    # Computing current true system state
    theta_k = theta_k_1 + ts*(omega_k_1)
    omega_k = omega_k_1 + ts*(-(g_k_1/L_k_1)*np.sin(theta_k_1)) 

    # Computing current true system parameters
    g_k = g_k_1
    L_k = L_k_1
                                        
    x_k_true = np.reshape(np.array([theta_k, omega_k, g_k, L_k]),(4,1))
                                        
    # Return statement
    return x_k_true 

# Defining the Simple Pendulum Discrete-Time Nonlinear Measurement Function
def SimplePendulum_h1(x_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        
    Returns:
        y_k_true (numpy.array): Current true system measurement
        y_k (numpy.array): Current noise corrupted system measurement        
    """
    
    # Simulation parameters
    
    # Getting individual state components
    theta_k_1 = x_k_1[0,0]
    omega_k_1 = x_k_1[1,0]
    
    # Computing current true system state
    omega_k = omega_k_1 
    
    # Computing current true system measurement
    y_k_true = omega_k
    y_k_true = np.reshape(y_k_true,(1,1))
                                        
    # Return statement
    return y_k_true 


#----------------------------------------------------------Defining System Model Jacobian----------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
# Defining the Simple Pendulum Discrete-Time Linearized System Function
def SimplePendulum_F(x_k_1, u_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        u_k_1 (numpy.array): Previous system measurement
        
    Returns:
        F (numpy.array): System linearized Dynamics Jacobian      
    """
        
    # Getting individual state components
    theta_k_1 = x_k_1[0,0]
    omega_k_1 = x_k_1[1,0]

    # Getting the parameter components
    g_k_1 = x_k_1[2,0]
    L_k_1 = x_k_1[3,0]
    
    # Computing System State Jacobian
    F = np.reshape(np.array([[0, 1, 0, 0],[-(g_k_1/L_k_1)*np.cos(theta_k_1), 0, -(1/L_k_1)*np.sin(theta_k_1), (g_k_1/L_k_1**2)*np.sin(theta_k_1)], [0, 0, 1, 0], [0, 0, 0, 1]]),(4,4))
    
                                        
    # Return statement
    return F 

# Defining the Simple Pendulum Discrete-Time Linearized Measurement Function
def SimplePendulum_H(x_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        
    Returns:
        H (numpy.array): System linearized Measurement Jacobian      
    """
        
    # Getting individual state components
    theta_k_1 = x_k_1[0,0]
    omega_k_1 = x_k_1[1,0]
    
    # Computing System State Jacobian
    H = np.reshape(np.array([0, 1, 0, 0]),(1,4))
    
                                        
    # Return statement
    return H 

#------------------------------------------------------------Simulation of True System-------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
## System Setup

# True initial states of the system
theta_ini_deg_true = 10.
omega_ini_rads_true = 0.

# True Parameters of the System
g_true = 9.86
L_true = 1.0

# Input to the system
u = 0

## Time Simulation Control Setup

# Discrete Time-Step
ts = 0.001

# Start Time
T_start = 0.

# Final Time
T_final = 30.

## Plotting Setup

# Plot Size parameters
Plot_Width = 15
Plot_Height = 10

## Basic Computation

# Convert initial theta to radians
theta_ini_rad_true = float(np.radians(theta_ini_deg_true))

# Create the true initial state vector
x_ini_true = np.reshape(np.array([theta_ini_rad_true, omega_ini_rads_true]), (2,1))

# Create the input vector
u_k = np.reshape(np.array([u]), (1,1)) 

# Create time vector
time_vector = np.arange(T_start, T_final+ts, ts)

## System Simulation

# Initializing system true state array to store time evolution
x_sim_nonlinear_true = x_ini_true

# Initializing system noisy state array to store time evolution
x_sim_nonlinear_noisy = x_ini_true

# Initializing system true measurement array to store time evolution
y_sim_nonlinear_true = np.reshape(x_ini_true[1,0],(1,1))

# Initializing system noisy measurement array to store time evolution
y_sim_nonlinear_noisy = np.reshape(x_ini_true[1,0],(1,1))

# FOR LOOP: For each discrete time-step
for ii in range(time_vector.shape[0]):
    
    # Computing next state of the Linear system
    x_k_true, x_k = SimplePendulum_f(np.reshape(x_sim_nonlinear_true[:,ii],(2,1)), u_k)
    
    y_k_true, y_k = SimplePendulum_h(np.reshape(x_k_true,(2,1)))   
    
    # Storing the states and measurements
    x_sim_nonlinear_true = np.hstack((x_sim_nonlinear_true, x_k_true))
    x_sim_nonlinear_noisy = np.hstack((x_sim_nonlinear_noisy, x_k))
    y_sim_nonlinear_true = np.hstack((y_sim_nonlinear_true, y_k_true))
    y_sim_nonlinear_noisy = np.hstack((y_sim_nonlinear_noisy, y_k))

## Simulation Plotting
# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting True States of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, x_sim_nonlinear_true[:,:-1].transpose(), label=[r'$\theta$ $(rads)$', r'$\omega = y$ $(rads/s)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True States', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting Noisy States of Nonlinear System
plt.subplot(222)
plt.plot(time_vector, x_sim_nonlinear_noisy[:,:-1].transpose(), label=[r'$\theta$ $(rads)$', r'$\omega = y$ $(rads/s)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - Noisy States', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True Parameter - g -  of Nonlinear System
plt.subplot(223)
plt.plot(time_vector, g_true*np.ones((np.shape(time_vector)[0],1)), label=r'$g$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(g)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True Parameter '+ r'$g$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True Parameter - L -  of Nonlinear System
plt.subplot(224)
plt.plot(time_vector, L_true*np.ones((np.shape(time_vector)[0],1)), label=r'$L$ $(m)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(L)$', fontsize=12)
plt.title('Simple Pendulum Linear System  True Parameter '+ r'$L$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

#plt.show()

#------------------------------------------------------------Simulation of Filter Step ------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
## System Model Setup

# Model parameters
g_model = 9.
L_model = 0.95

# Model input
u_model = 0

## Filter Setup

# Initial Filter stae mean/covariance
theta_ini_model = 9.
omega_ini_model = 0.5

g_ini_model = 1.
L_ini_model = 0.1

P_model = 1

# Filter constant parameters
n = 4  # Dimension of states of the system
m = 1  # Dimension of measurements of the system

GF_Type = 2  # GF_Type: 1 - Gauss-Hermite ; 2 - Spherical Cubature 
p = 5  # Order of Hermite Polynomial

alpha = 0.1 # Controls spread of Sigma Points about mean
k = 0.9  # Controls spread of Sigma Points about mean
beta = 2  # Helps to incorporate prior information abou thwe non-Gaussian

# Filter process/measurement noise covariances
Q_model_EKFS = 1
Q_params_EKFS = 10
R_model_EKFS = 0.01

Q_model_UKFS = 100
Q_params_UKFS = 10
R_model_UKFS = 0.01

Q_model_GFS = 10
Q_params_GFS = 1
R_model_GFS = 0.1

## Model Computations

# Create the input vector
u_k_model = np.reshape(np.array([u_model]), (1,1)) 

## Filter Computations

# Convert initial theta to radians
theta_ini_rad_model = float(np.radians(theta_ini_model))

# Creating initial Filter mean vector
m_ini_model = np.reshape(np.array([theta_ini_rad_model, omega_ini_model, g_ini_model, L_ini_model]), (4,1))

# Creating initial Filter state covariance matrix
P_ini_model = P_model*np.eye(4)

# Create the model Q and R matrices
Q_d_EKFS = np.reshape(np.array([[Q_model_EKFS, 0, 0, 0],[0, Q_model_EKFS, 0, 0],[0, 0, Q_params_EKFS, 0],[0, 0, 0, Q_params_EKFS]]), (4,4))
R_d_EKFS = np.reshape(np.array([R_model_EKFS]), (1,1))

Q_d_UKFS = np.reshape(np.array([[Q_model_UKFS, 0, 0, 0],[0, Q_model_UKFS, 0, 0],[0, 0, Q_params_UKFS, 0],[0, 0, 0, Q_params_UKFS]]), (4,4))
R_d_UKFS = np.reshape(np.array([R_model_UKFS]), (1,1))

Q_d_GFS = np.reshape(np.array([[Q_model_GFS, 0, 0, 0],[0, Q_model_GFS, 0, 0],[0, 0, Q_params_GFS, 0],[0, 0, 0, Q_params_GFS]]), (4,4))
R_d_GFS = np.reshape(np.array([R_model_GFS]), (1,1))

## Creating Object for Noninear Model
SimplePedulum_Nonlinear_EKF = EKFS(SimplePendulum_f1, SimplePendulum_F, SimplePendulum_h1, SimplePendulum_H, m_ini_model, P_ini_model, Q_d_EKFS, R_d_EKFS)
SimplePedulum_Nonlinear_UKF = UKFS(SimplePendulum_f1, SimplePendulum_h1, n, m, alpha, k, beta, m_ini_model, P_ini_model, Q_d_UKFS, R_d_UKFS)
SimplePedulum_Nonlinear_GF = GFS(GF_Type, SimplePendulum_f1, SimplePendulum_h1, n, m, m_ini_model, P_ini_model, Q_d_GFS, R_d_GFS, p)

## Filter Time Evolution
# Initializing model filter state array to store time evolution
x_model_nonlinear_filter_EKFS = m_ini_model

x_model_nonlinear_filter_UKFS = m_ini_model

x_model_nonlinear_filter_GFS = m_ini_model


# FOR LOOP: For each discrete time-step
for ii in range(y_sim_nonlinear_noisy.shape[1]):
    
    ## For measurements coming from Nonlinear System
    
    # Extended Kalman Filter: Predict Step    
    m_k_, P_k_ = SimplePedulum_Nonlinear_EKF.Extended_Kalman_Predict(u_k_model)
    
    # Extended Kalman Filter: Update Step
    v_k, S_k, K_k = SimplePedulum_Nonlinear_EKF.Extended_Kalman_Update(np.reshape(y_sim_nonlinear_noisy[:,ii], (1,1)), m_k_, P_k_)    
    
    # Storing the Filtered states
    x_k_filter = SimplePedulum_Nonlinear_EKF.m_k
    x_model_nonlinear_filter_EKFS = np.hstack((x_model_nonlinear_filter_EKFS, x_k_filter))


# FOR LOOP: For each discrete time-step
for ii in range(y_sim_nonlinear_noisy.shape[1]):
    
    ## For measurements coming from Nonlinear System
    
    # Extended Kalman Filter: Predict Step    
    m_k_, P_k_, D_k = SimplePedulum_Nonlinear_UKF.Unscented_Kalman_Predict(u_k_model)
    
    # Extended Kalman Filter: Update Step
    mu_k, S_k, C_k, K_k = SimplePedulum_Nonlinear_UKF.Unscented_Kalman_Update(np.reshape(y_sim_nonlinear_noisy[:,ii], (1,1)), m_k_, P_k_)    
    
    # Storing the Filtered states
    x_k_filter = SimplePedulum_Nonlinear_UKF.m_k
    x_model_nonlinear_filter_UKFS = np.hstack((x_model_nonlinear_filter_UKFS, x_k_filter))


# FOR LOOP: For each discrete time-step
for ii in range(y_sim_nonlinear_noisy.shape[1]):
    
    ## For measurements coming from Nonlinear System
    
    # Extended Kalman Filter: Predict Step    
    m_k_, P_k_, D_k = SimplePedulum_Nonlinear_GF.Gaussian_Predict(u_k_model)
    
    # Extended Kalman Filter: Update Step
    mu_k, S_k, C_k, K_k = SimplePedulum_Nonlinear_GF.Gaussian_Update(np.reshape(y_sim_nonlinear_noisy[:,ii], (1,1)), m_k_, P_k_)    
    
    # Storing the Filtered states
    x_k_filter = SimplePedulum_Nonlinear_GF.m_k
    x_model_nonlinear_filter_GFS = np.hstack((x_model_nonlinear_filter_GFS, x_k_filter))

## Filter Simulation Plotting

# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting True vs. Filtered States of Nonlinear System
plt.subplot(311)
plt.plot(time_vector, x_sim_nonlinear_true[:,:-1].transpose(), linestyle='-', linewidth=3, label=[r'$\theta_{true}$ $(rads)$', r'$\omega_{true}$ $(rads/s)$'])
plt.plot(time_vector, x_model_nonlinear_filter_EKFS[0:2,:-2].transpose(), linestyle='--', linewidth=1, label=[r'$\theta_{filter}$ $(rads)$', r'$\omega_{filter}$ $(rads/s)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Filtered States - EKF', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System
plt.subplot(312)
plt.plot(time_vector, g_true*np.ones((np.shape(time_vector)[0],1)), label=r'$g_{true}$ $(m/s^{2})$')
plt.plot(time_vector, x_model_nonlinear_filter_EKFS[2,:-2].transpose(), linestyle='--', linewidth=1, label=r'$g_{filter}$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(g)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Filtered Parameter ' + r'$g$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.subplot(313)
plt.plot(time_vector, L_true*np.ones((np.shape(time_vector)[0],1)), label=r'$L_{true}$ $(m)$')
plt.plot(time_vector, x_model_nonlinear_filter_EKFS[3,:-2].transpose(), linestyle='--', linewidth=1, label=r'$L_{filter}$ $(m)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(L)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Filtered Parameter ' + r'$L$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


plt.figure()

# Plotting True vs. Filtered States of Nonlinear System
plt.subplot(311)
plt.plot(time_vector, x_sim_nonlinear_true[:,:-1].transpose(), linestyle='-', linewidth=3, label=[r'$\theta_{true}$ $(rads)$', r'$\omega_{true}$ $(rads/s)$'])
plt.plot(time_vector, x_model_nonlinear_filter_UKFS[0:2,:-2].transpose(), linestyle='--', linewidth=1, label=[r'$\theta_{filter}$ $(rads)$', r'$\omega_{filter}$ $(rads/s)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Filtered States - UKF', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System
plt.subplot(312)
plt.plot(time_vector, g_true*np.ones((np.shape(time_vector)[0],1)), label=r'$g_{true}$ $(m/s^{2})$')
plt.plot(time_vector, x_model_nonlinear_filter_UKFS[2,:-2].transpose(), linestyle='--', linewidth=1, label=r'$g_{filter}$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(g)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Filtered Parameter ' + r'$g$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.subplot(313)
plt.plot(time_vector, L_true*np.ones((np.shape(time_vector)[0],1)), label=r'$L_{true}$ $(m)$')
plt.plot(time_vector, x_model_nonlinear_filter_UKFS[3,:-2].transpose(), linestyle='--', linewidth=1, label=r'$L_{filter}$ $(m)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(L)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Filtered Parameter ' + r'$L$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


plt.figure()

# Plotting True vs. Filtered States of Nonlinear System
plt.subplot(311)
plt.plot(time_vector, x_sim_nonlinear_true[:,:-1].transpose(), linestyle='-', linewidth=3, label=[r'$\theta_{true}$ $(rads)$', r'$\omega_{true}$ $(rads/s)$'])
plt.plot(time_vector, x_model_nonlinear_filter_GFS[0:2,:-2].transpose(), linestyle='--', linewidth=1, label=[r'$\theta_{filter}$ $(rads)$', r'$\omega_{filter}$ $(rads/s)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Filtered States - GF', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System
plt.subplot(312)
plt.plot(time_vector, g_true*np.ones((np.shape(time_vector)[0],1)), label=r'$g_{true}$ $(m/s^{2})$')
plt.plot(time_vector, x_model_nonlinear_filter_GFS[2,:-2].transpose(), linestyle='--', linewidth=1, label=r'$g_{filter}$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(g)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Filtered Parameter ' + r'$g$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.subplot(313)
plt.plot(time_vector, L_true*np.ones((np.shape(time_vector)[0],1)), label=r'$L_{true}$ $(m)$')
plt.plot(time_vector, x_model_nonlinear_filter_GFS[3,:-2].transpose(), linestyle='--', linewidth=1, label=r'$L_{filter}$ $(m)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(L)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Filtered Parameter ' + r'$L$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.show()

#-----------------------------------------------------------Simulation of Smoother Step -----------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
