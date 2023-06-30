import os
import sys
import numpy as np
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join('./bayesian-filters-smoothers/bayesian_filters_smoothers'))

print(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)

import bayesian_filters_smoothers as bfs

from bayesian_filters_smoothers import Kalman_Filter_Smoother as KFS

""" a=bfs.addnum(10,10)

print(a)

a=bfs.subnum(10,10)

print(a) """


############################################################################################################################################################

def SimplePendulum_Linear_DiscreteTime_System(x_k_1, u_k_1):
    """Provides discrete time dynamics for a linear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        u_k_1 (numpy.array): Previous system measurement
        g (float): Acceleration due to gravity
        L (float): Length of the pendulum
        ts (float): Discrete time-step
        Q (numpy.array): Process noise true covariance matrix
        R (numpy.array): Measurement noise true covariance matrix
        
    Returns:
        x_k_true (numpy.array): Current true system state
        y_k_true (numpy.array): Current true system measurement
        y_k (numpy.array): Current noise corrupted system measurement        
    """
    # Simulation parameters
    g = 9.86
    L = 1.
    Q = 0.01
    R = 0.1
    
    # Create the true Q and R matrices
    Q = np.reshape(np.array([[Q, 0],[0, Q]]), (2,2))
    R = np.reshape(np.array([R]), (1,1))
    
    # Creating Continuous Time System matrices
    A = np.reshape(np.array([[0,1],[-(g/L),0]]), (2,2))
    B = np.reshape(np.array([0,0]), (2,1))
    C = np.reshape(np.array([0,1]), (1,2))
    
    # Creating Discrete Time System matrices
    A_d = np.eye(2) + (ts*A)
    B_d = ts*B
    C_d = C
    
    # Computing current true system state
    x_k_true = (np.dot(A_d, x_k_1) + np.dot(B_d, u_k_1))
    x_k_true = np.reshape(x_k_true, (2,1))
    
    # Computing current true system measurement
    y_k_true = np.dot(C_d, x_k_true)
    
    # Computing Process/Measurement Noise
    q_k_1 = np.random.multivariate_normal(np.array([0,0]), Q)
    q_k_1 = np.reshape(q_k_1, (2,1))
    
    r_k = np.random.multivariate_normal(np.array([0]), R)
                                        
    # Computing noise corrupted system state/measurement
    x_k = x_k_true + q_k_1
    y_k = y_k_true + r_k
                                        
    # Return statement
    return x_k_true, y_k_true, x_k, y_k  
                                        
# Defining the Simple Pendulum Discrete-Time Nonlinear System Function
def SimplePendulum_Nonlinear_DiscreteTime_System(x_k_1, u_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        u_k_1 (numpy.array): Previous system measurement
        g (float): Acceleration due to gravity
        L (float): Length of the pendulum
        ts (float): Discrete time-step
        Q (numpy.array): Process noise true covariance matrix
        R (numpy.array): Measurement noise true covariance matrix
        
    Returns:
        x_k_true (numpy.array): Current true system state
        y_k_true (numpy.array): Current true system measurement
        x_k (numpy.array): Current noise corrupted system state
        y_k (numpy.array): Current noise corrupted system measurement        
    """
    
    # Simulation parameters
    g = 9.86
    L = 1.
    Q = 0.01
    R = 0.1
    
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
    
    # Computing current true system measurement
    y_k_true = omega_k_1
    y_k_true = np.reshape(y_k_true,(1,1))
    
    # Computing Process/Measurement Noise
    q_k_1 = np.random.multivariate_normal(np.array([0,0]), Q)
    q_k_1 = np.reshape(q_k_1, (2,1))
                                        
    r_k = np.random.multivariate_normal(np.array([0]), R)
                                        
    # Computing noise corrupted system state/measurement
    x_k = x_k_true + q_k_1
    y_k = y_k_true + r_k
                                        
    # Return statement
    return x_k_true, y_k_true, x_k, y_k 

## System Setup

# True initial states of the system
theta_ini_deg_true = 10.
omega_ini_rads_true = 0.

# Input to the system
u = 0

## Time Simulation Control Setup

# Discrete Time-Step
ts = 0.001

# Start Time
T_start = 0.

# Final Time
T_final = 10.

## Plotting Setup

# Plot Size parameters
Plot_Width = 15
Plot_Height = 10

# Convert initial theta to radians
theta_ini_rad_true = float(np.radians(theta_ini_deg_true))

# Create the true initial state vector
x_ini_true = np.reshape(np.array([theta_ini_rad_true, omega_ini_rads_true]), (2,1))

# Create the input vector
u_k = np.reshape(np.array([u]), (1,1)) 

# Create time vector
time_vector = np.arange(T_start, T_final+ts, ts)

# Initializing system true state array to store time evolution
x_sim_linear_true = x_ini_true
x_sim_nonlinear_true = x_ini_true

# Initializing system noisy state array to store time evolution
x_sim_linear_noisy = x_ini_true
x_sim_nonlinear_noisy = x_ini_true

# Initializing system true measurement array to store time evolution
y_sim_linear_true = np.reshape(x_ini_true[1,0],(1,1))
y_sim_nonlinear_true = np.reshape(x_ini_true[1,0],(1,1))

# Initializing system noisy measurement array to store time evolution
y_sim_linear_noisy = np.reshape(x_ini_true[1,0],(1,1))
y_sim_nonlinear_noisy = np.reshape(x_ini_true[1,0],(1,1))

# FOR LOOP: For each discrete time-step
for ii in range(time_vector.shape[0]):
    
    # Computing next state of the Linear system
    x_k_true, y_k_true, x_k, y_k = SimplePendulum_Linear_DiscreteTime_System(np.reshape(x_sim_linear_true[:,ii],(2,1)), u_k)
    
    # Storing the states and measurements
    x_sim_linear_true = np.hstack((x_sim_linear_true, x_k_true))
    x_sim_linear_noisy = np.hstack((x_sim_linear_noisy, x_k))
    y_sim_linear_true = np.hstack((y_sim_linear_true, y_k_true))
    y_sim_linear_noisy = np.hstack((y_sim_linear_noisy, y_k))
    
    # Computing next state of the Nonlinear system
    x_k_true, y_k_true, x_k, y_k = SimplePendulum_Nonlinear_DiscreteTime_System(np.reshape(x_sim_nonlinear_true[:,ii],(2,1)), u_k)
    
    # Storing the states and measurements
    x_sim_nonlinear_true = np.hstack((x_sim_nonlinear_true, x_k_true))
    x_sim_nonlinear_noisy = np.hstack((x_sim_nonlinear_noisy, x_k))
    y_sim_nonlinear_true = np.hstack((y_sim_nonlinear_true, y_k_true))
    y_sim_nonlinear_noisy = np.hstack((y_sim_nonlinear_noisy, y_k))


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

P_model = 1

# Filter process/measurement noise covariances
Q_model = 0
R_model = 0.1

## Model Computations

# Creating Continuous Time Model matrices
A = np.reshape(np.array([[0,1],[-(g_model/L_model),0]]), (2,2))
B = np.reshape(np.array([0,0]), (2,1))
C = np.reshape(np.array([0,1]), (1,2))

# Creating Discrete Time Model matrices
A_d = np.eye(2) + (ts*A)
B_d = ts*B
C_d = C

## Create the input vector list

# Create input vector
u_k_model = np.reshape(np.array([u_model]), (1,1)) 

# Create the input list
u_k_model_list = [] 

for ii in range(y_sim_linear_noisy.shape[1]):
    
    u_k_model_list.append(u_k_model)
    
## Create the measurement vector list

# For measurements from Linear System
y_sim_linear_noisy_list = []

for ii in range(y_sim_linear_noisy.shape[1]):
    
    y_sim_linear_noisy_list.append(np.reshape(y_sim_linear_noisy[:,ii], (1,1)))

# For measurements from Nonlinear System
y_sim_nonlinear_noisy_list = []

for ii in range(y_sim_nonlinear_noisy.shape[1]):
    
    y_sim_nonlinear_noisy_list.append(np.reshape(y_sim_nonlinear_noisy[:,ii], (1,1)))

## Filter Computations

# Convert initial theta to radians
theta_ini_rad_model = float(np.radians(theta_ini_model))

# Creating initial Filter mean vector
m_ini_model = np.reshape(np.array([theta_ini_rad_model, omega_ini_model]), (2,1))

# Creating initial Filter state covariance matrix
P_ini_model = P_model*np.eye(2)

# Create the model Q and R matrices
Q_d = np.reshape(np.array([[Q_model, 0],[0, Q_model]]), (2,2))
R_d = np.reshape(np.array([R_model]), (1,1))

# Creating Object for Linear Model
SimplePedulum_Linear_KS = KFS(A_d, B_d, C_d, m_ini_model, P_ini_model, Q_d, R_d)

# Creating Object for Nonlinear Model
SimplePedulum_Nonlinear_KS = KFS(A_d, B_d, C_d, m_ini_model, P_ini_model, Q_d, R_d)


## For measurements coming from Linear System

# Kalman Smoother  
G_k_list, m_k_s_list, P_k_s_list = SimplePedulum_Linear_KS.Kalman_Smoother(u_k_model_list, y_sim_linear_noisy_list)   

# Storing the Filtered states
for ii in range(len(m_k_s_list)):
    
    if (ii == 0):
        
        x_model_linear_smoother = m_k_s_list[ii]
        
    else:
        
        x_model_linear_smoother = np.hstack((x_model_linear_smoother, m_k_s_list[ii]))

## For measurements coming from Nonlinear System

# Kalman Smoother    
G_k_list, m_k_s_list, P_k_s_list = SimplePedulum_Nonlinear_KS.Kalman_Smoother(u_k_model_list, y_sim_nonlinear_noisy_list) 

# Storing the Filtered states
for ii in range(len(m_k_s_list)):
    
    if (ii == 0):
        
        x_model_nonlinear_smoother = m_k_s_list[ii]
        
    else:
        
        x_model_nonlinear_smoother = np.hstack((x_model_nonlinear_smoother, m_k_s_list[ii]))


# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting True vs. Filtered States of Linear System
plt.subplot(211)
plt.plot(time_vector, x_sim_linear_true[:,:-1].transpose(), linestyle='-', linewidth=3, label=[r'$\theta_{true}$ $(rads)$', r'$\omega_{true}$ $(rads/s)$'])
plt.plot(time_vector, x_model_linear_smoother[:,:-1].transpose(), linestyle='--', linewidth=1, label=[r'$\theta_{smoother}$ $(rads)$', r'$\omega_{smoother}$ $(rads/s)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Simple Pendulum Linear System - True vs. Smoothed States', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True)


# Plotting True vs. Filtered States of Nonlinear System
plt.subplot(212)
plt.plot(time_vector, x_sim_nonlinear_true[:,:-1].transpose(), linestyle='-', linewidth=3, label=[r'$\theta_{true}$ $(rads)$', r'$\omega_{true}$ $(rads/s)$'])
plt.plot(time_vector, x_model_nonlinear_smoother[:,:-1].transpose(), linestyle='--', linewidth=1, label=[r'$\theta_{smoother}$ $(rads)$', r'$\omega_{smoother}$ $(rads/s)$'])
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Simple Pendulum Nonlinear System - True vs. Smoothed States', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True)

plt.show()