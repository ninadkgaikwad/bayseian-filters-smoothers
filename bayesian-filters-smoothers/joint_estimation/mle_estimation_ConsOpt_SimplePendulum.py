## Importing external modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from casadi import *


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
    F = np.reshape(np.array([[0, 1, 0, 0],[-(g_k_1/L_k_1)*np.cos(theta_k_1), 0, -(1/L_k_1)*np.sin(theta_k_1), (g_k_1/L_k_1**2)*np.sin(theta_k_1)], [0, 0, 0, 0], [0, 0, 0, 0]]),(4,4))
    
                                        
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
T_final = 2.

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


#----------------------------------------------------------------Batch Estimation -----------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#

## Initial Setup

# State/Parameter/Output Dimensions
State_n = 2
Parameter_n = 2
Output_n = 1

# Initial Filter stae mean/covariance
theta_ini_model = 9.
omega_ini_model = 0.5

g_ini_model = 9.86
L_ini_model = 1.0

# State Covariance
P_model = 1

# Filter process/measurement noise covariances
Q_model = 0.01
Q_params = 0.01
R_model = 0.01

# Creating Infinity
Infinity = np.inf

## Getting total time steps
N = y_sim_nonlinear_noisy.shape[1]

## Creating Optimization Variables

# State Variables
x1_l = SX.sym('x1_l',N+1,1)
x2_l = SX.sym('x2_l',N+1,1)

# Parameter Variables
g = SX.sym('g',1,1)
L = SX.sym('L',1,1)

# System Matrix
A_matrix = SX.sym('A_matrix',State_n,State_n)

A_matrix[0,0] = 1
A_matrix[0,1] = ts
A_matrix[1,0] = -(g/L)*ts
A_matrix[1,1] = 1

# Other Variables
S_l = SX.sym('S_l',N,1)
e_l = SX.sym('e_l',N,1)
P_l = SX.sym('P_l',(State_n**State_n)*(N+1),1)

# System Constants
C_matrix = DM(1,State_n)
C_matrix[:,:] = np.reshape(np.array([0,1]), (1,State_n))

R_matrix = DM(Output_n,Output_n)
R_matrix[:,:] = np.reshape(R_model*np.eye(Output_n), (Output_n,Output_n))

Q_matrix = DM(State_n,State_n)
Q_matrix[:,:] = np.reshape(Q_model*np.eye(State_n), (State_n,State_n))

## Constructing the Cost Function

# Cost Function Development
CostFunction = 0

## Constructing the Constraints

# Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
x1_lb = []
x1_ub = []

x2_lb = []
x2_ub = []

g_lb = [9.]
g_ub = [10.]

L_lb = [0.9]
L_ub = [1.1]

S_lb = []
S_ub = []

e_lb = []
e_ub = []

P_lb = []
P_ub = []


Eq_x_lb = []
Eq_P_lb = []
Eq_S_lb = []
Eq_e_lb = []

Eq_x_ub = []
Eq_P_ub = []
Eq_S_ub = []
Eq_e_ub = []

Eq_x = []
Eq_P = []
Eq_S = []
Eq_e = []

# FOR LOOP: For each time step
for ii in range(N):   
    
    # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
    CostFunction += e_l[ii]**2 * (1/S_l[ii]) + log(S_l[ii])
        
    ## State/Covariance Equations - Formulation
    
    # Creating State Vector
    x_k_1 = SX.sym('x_k_1',State_n,1)
    x_k = SX.sym('x_k',State_n,1)
    
    x_k_1[0,0] = x1_l[ii+1]
    x_k_1[1,0] = x2_l[ii+1]
    
    x_k[0,0] = x1_l[ii]
    x_k[1,0] = x2_l[ii]
    
    # Creating P matrix
    P_matrix_k = SX.sym('P_matrix_k', State_n, State_n)
    P_matrix_k_1 = SX.sym('P_matrix_k_1', State_n, State_n)
    
    P_matrix_k_1[0,0] = P_l[(ii+1)*State_n**2]
    P_matrix_k_1[1,0] = P_l[((ii+1)*State_n**2)+1]
    P_matrix_k_1[0,1] = P_l[((ii+1)*State_n**2)+2]
    P_matrix_k_1[1,1] = P_l[((ii+1)*State_n**2)+3]
    
    P_matrix_k[0,0] = P_l[ii*State_n**2]
    P_matrix_k[1,0] = P_l[(ii*State_n**2)+1]
    P_matrix_k[0,1] = P_l[(ii*State_n**2)+2]
    P_matrix_k[1,1] = P_l[(ii*State_n**2)+3]
    
    # State Equation
    x_Eq = -x_k_1 + A_matrix @ (x_k + P_matrix_k @ C_matrix.T @ (1/S_l[ii]) @ e_l[ii])

    # Covariance Equation
    P_Eq = -P_matrix_k_1 + A_matrix @ (P_matrix_k - P_matrix_k @ C_matrix.T @ (1/S_l[ii]) @ C_matrix @ P_matrix_k) @ A_matrix.T + Q_matrix

    ## Filter Update Equations
    
    # S_k Equation
    S_Eq = -S_l[ii] + (C_matrix @ P_matrix_k @ C_matrix.T) + R_matrix
    
    # e_k Equation
    e_Eq = -e_l[ii] + y_sim_nonlinear_noisy[0,ii] - (C_matrix @ x_k)

    # Adding current equations to Equation List
    Eq_x += [x_Eq[0,0], x_Eq[1,0]]
    
    Eq_P += [P_Eq[0,0], P_Eq[1,0], P_Eq[0,1], P_Eq[1,1]]
    
    Eq_S += [S_Eq]
    
    Eq_e += [e_Eq]

    # Adding Equation Bounds
    Eq_x_lb += [0, 0]
    Eq_x_ub += [0, 0]
    
    Eq_P_lb += [0, 0, 0, 0]
    Eq_P_ub += [0, 0, 0, 0]
    
    Eq_S_lb += [0]
    Eq_S_ub += [0]
    
    Eq_e_lb += [0]
    Eq_e_ub += [0]

    # Adding Variable Bounds
    x1_lb += [-Infinity]
    x1_ub += [Infinity]

    x2_lb += [-Infinity]
    x2_ub += [Infinity]
    
    P_lb += [-Infinity, -Infinity, -Infinity, -Infinity]
    P_ub += [Infinity, Infinity, Infinity, Infinity]

    S_lb += [0.0001]
    S_ub += [Infinity]

    e_lb += [-Infinity]
    e_ub += [Infinity]

## Adding Variable Bounds - For (N+1) Variables
x1_lb += [-Infinity]
x1_ub += [Infinity]

x2_lb += [-Infinity]
x2_ub += [Infinity]

P_lb += [-Infinity, -Infinity, -Infinity, -Infinity]
P_ub += [Infinity, Infinity, Infinity, Infinity]    

## Constructing NLP Problem

# Creating Optimization Variable: x
x = vcat([g, L, x1_l, x2_l, P_l, S_l, e_l])

# Creating Cost Function: J
J = CostFunction

# Creating Constraints: g
g = vertcat(*Eq_x, *Eq_P, *Eq_S, *Eq_e)

# Creating NLP Problem
NLP_Problem = {'f': J, 'x': x, 'g': g}

## Constructiong NLP Solver
NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

## Solving the NLP Problem

# Creating Initial Variables
x1_l_ini = (float(np.radians(theta_ini_model))*np.ones((N+1,))).tolist()
x2_l_ini = (omega_ini_model*np.ones((N+1,))).tolist()
g_ini = (g_ini_model*np.ones((1,))).tolist()
L_ini = (L_ini_model*np.ones((1,))).tolist()
P_l_ini = [1,0,0,1]*(N+1)
S_l_ini = (0.001*np.ones((N,))).tolist()
e_l_ini = np.zeros((N,)).tolist()

x_initial = vertcat(*g_ini, *L_ini, *x1_l_ini, *x2_l_ini, *P_l_ini, *S_l_ini, *e_l_ini)

# Creating Lower/Upper bounds on Variables and Equations
x_lb = vertcat(*g_lb, *L_lb, *x1_lb, *x2_lb, *P_lb, *S_lb, *e_lb)

x_ub = vertcat(*g_ub, *L_ub, *x1_ub, *x2_ub, *P_ub, *S_ub, *e_ub)

G_lb = vertcat(*Eq_x_lb, *Eq_P_lb, *Eq_S_lb, *Eq_e_lb)

G_ub = vertcat(*Eq_x_ub, *Eq_P_ub, *Eq_S_ub, *Eq_e_ub)

# Solving NLP Problem
NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

#----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#

## Getting the Solutions
NLP_Sol = NLP_Solution['x'].full().flatten()

g_sol = NLP_Sol[0]
L_sol = NLP_Sol[1]
x1_sol = NLP_Sol[2:(N+1)+2]
x2_sol = NLP_Sol[((N+1)+2):(2*(N+1)+2)]
P_sol = NLP_Sol[(2*(N+1)+2):(6*(N+1)+2)]
S_sol = NLP_Sol[(6*(N+1)+2):(6*(N+1)+2+N)]
e_sol = NLP_Sol[(6*(N+1)+2+N):(6*(N+1)+2+2*N)]

## Simulation Plotting
# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting  States 
plt.plot(time_vector, x1_sol[0:-2], label=r'$\theta$ $(rads)$')
plt.plot(time_vector, x2_sol[0:-2], label=r'$\omega = y$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('States '+ r'$(x)$', fontsize=12)
plt.title('Simple Pendulum States - NLP Solution ' + r'x', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting Figures
plt.figure()

# Plotting  Parameters - g -  of Nonlinear System
plt.subplot(221)
plt.plot(time_vector, g_sol*np.ones((len(time_vector),1)), label=r'$g$ $(m/s^{2})$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(g)$', fontsize=12)
plt.title('Simple Pendulum Parameter - NLP Solution '+ r'$g$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting  Parameters - L -  of Nonlinear System
plt.subplot(222)
plt.plot(time_vector, L_sol*np.ones((len(time_vector),1)), label=r'$L$ $(m)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(L)$', fontsize=12)
plt.title('Simple Pendulum Parameter - NLP Solution '+ r'$L$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


# Plotting  Variables - S -  of Nonlinear System
plt.subplot(223)
plt.plot(time_vector, S_sol[0:-1], label=r'$S$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(S)$', fontsize=12)
plt.title('Simple Pendulum Variable - NLP Solution '+ r'$S$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


# Plotting  Variables - e -  of Nonlinear System
plt.subplot(224)
plt.plot(time_vector, e_sol[0:-1], label=r'$e$ $(rads/s)$')
plt.xlabel('Time ' + r'$(sec)$', fontsize=12)
plt.ylabel('Variable '+ r'$(e)$', fontsize=12)
plt.title('Simple Pendulum Variable - NLP Solution '+ r'$e$', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)


