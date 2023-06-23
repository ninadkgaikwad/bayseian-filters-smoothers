# -*- coding: utf-8 -*-

# Importing external modules
import os
import math as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing internal modules
from .utils import utils 

"""Bayesian Filters and Smoothers

This module implements the following Bayesian Filters and Smoothers:

    Kalman Filter/Smoother
    Extended Kalman Filter/Smoother
    Unscented Kalman Filter/Smoother
    Gaussian Filter/Smoother
    Particle Filter/Smoother

Classes:

    Kalman_Filter_Smoother
    Extended_Kalman_Filter_Smoother

Functions:

    Kalman_Predict(self, u_k)
    Kalman_Update(self, y_k, m_k_, P_k_)
    Kalman_Smoother(self, u_k_list, y_k_list)
    Extended_Kalman_Predict(self, u_k)
    Extended_Kalman_Update(self, y_k, m_k_, P_k_)
    Extended_Kalman_Smoother(self, u_k_list, y_k_list)
    Unscented_Kalman_Predict(self, u_k)
    Unscented_Kalman_Update(self, y_k, m_k_, P_k_)
    Unscented_Kalman_Smoother(self, u_k_list, y_k_list)


Misc variables:

    __version__
"""

def addnum(num1,num2):
    """Adds two numbers.
    
    Args:
        num1 (int/float): The first number.
        num2 (int/float): The second number.

    Returns:
        num3 (int/float): Result of addition of num1 and num2.

    """
    num4 = utils.subnum(num1,num2)

    num3 = num2+num4
    return num3

class Kalman_Filter_Smoother:
    """Creates an object to handle Kalman Filtering and Smoothing for Linear Time Invariant (LTI) System

    Attributes:
        A (numpy.array): LTI system matrix
        B (numpy.array): LTI input matrix
        C (numpy.array): LTI output matrix
        m_k (numpy.array): State mean vector
        P_k (numpy.array): State covariance matrix
        Q (numpy.array): Process error covariance matrix
        R (numpy.array): Measurement error covariance matrix
    """
    
    def __init__(self, A, B, C, m_k, P_k, Q, R):
        """Initializes the instance of class Kalman_Filter_Smoother

        Args:
            A (numpy.array): LTI system matrix
            B (numpy.array): LTI input matrix
            C (numpy.array): LTI output matrix
            m_k (numpy.array): State mean vector
            P_k (numpy.array): State covariance matrix
            Q (numpy.array): Process error covariance matrix
            R (numpy.array): Measurement error covariance matrix
        """

        # Initializing the instance
        self.A = A
        self.B = B
        self.C = C
        self.m_k = m_k
        self.P_k = P_k
        self.Q = Q
        self.R = R

    def set_A(self, A):
        """Sets the A array for a linear time-varying (LTV) system
        
        Args:
            A (numpy.array): LTV system matrix
        """
        self.A = A

    def set_B(self, B):
        """Sets the B matrix for a linear time-varying (LTV) system
        
        Args:
            B (numpy.array): LTV input matrix
        """
        self.B = B

    def set_C(self, C):
        """Sets the C matrix for a linear time-varying (LTV) system
        
        Args:
            C (numpy.array): LTV output matrix
        """
        self.C = C

    def set_Q(self, Q):
        """Sets the Q matrix for the system
        
        Args:
            Q (numpy.array): Process error covariance matrix
        """
        self.Q = Q

    def set_R(self, R):
        """Sets the R matrix for the system
        
        Args:
            R (numpy.array): Measurement error covariance matrix
        """
        self.R = R

    def Kalman_Predict(self, u_k):
        """Performs the predict step of the Kalman Filter

        Args:
            u_k (numpy.array): Input to the system

        Returns:
            m_k_ (numpy.array): Predicted state mean vector
            P_k_ (numpy.array): Predicted state covariance matrix
        """

        # Computing predicted mean of the states
        m_k_ = np.dot(self.A, self.m_k) + np.dot(self.B, u_k)

        # Computing predicted covariance of the states
        P_k_ = np.dot(np.dot(self.A, self.P_k), self.A.transpose()) + self.Q

        # Return statement
        return m_k_, P_k_

    def Kalman_Update(self, y_k, m_k_, P_k_):
        """Performs the update step of the Kalman Filter

        Args:
            y_k (numpy.array): New measurement from the system
            m_k_ (numpy.array): Predicted state mean vector
            P_k_ (numpy.array): Predicted state covariance matrix

        Returns:
            v_k (numpy.array): Updated measurement error
            S_k (numpy.array): Updated measurement covariance matrix
            K_k (numpy.array): Updated Kalman Gain
        """

        # Computing measurement error
        v_k = y_k - (np.dot(self.C, m_k_))

        # Computing measurement covariance
        S_k = np.dot(np.dot(self.C, P_k_), self.C.transpose()) + self.R

        # Computing Kalman Gain
        K_k = np.dot(np.dot(P_k_, self.C.transpose()), np.linalg.inv(S_k))

        # Updating state mean vector
        self.m_k = m_k_ + np.dot(K_k, v_k)

        # Updating state covariance matix
        self.P_k = P_k_ - np.dot(np.dot(K_k, S_k), K_k.transpose())

        # Return statement
        return v_k, S_k, K_k

    def Kalman_Smoother(self, u_k_list, y_k_list):
        """Performs the smoothening step of the Kalman Smoother

        Args:
            u_k_list (list of numpy.array): List of inputs to the system
            y_k_list (list of numpy.array): List of measurements from the system

        Returns:
            G_k_list (list of numpy.array): List of Kalman Smoother Gain
            m_k_s (list of numpy.array): List of smoothed state vectors
            P_k_s (list of numpy.array): List of smoothed state covarivance matrices
        """

        # Computing length of data
        Data_Len = len(y_k_list)

        ## Performing Filtering step of the Kalman Filter
        m_k__list = []  # Initializing
        P_k__list = []  # Initializing
        m_k_list = []  # Initializing
        P_k_list = []  # Initializing

        # Initializing with initial State Mean and Covariance vector and matrix respectively
        m_k_list.append(self.m_k)
        P_k_list.append(self.P_k)

        # FOR LOOP: For each element in the y_k_list
        for ii in range(Data_Len):

            # Computing predict step of Kalman Filter
            m_k_, P_k_ = self.Kalman_Predict(self, u_k_list[ii])

            # Appending predicted mean/covariance lists
            m_k__list .append(m_k_)
            P_k__list .append(P_k_)

            # Computing update step of Kalman Filter
            self.Kalman_Update(self, y_k_list[ii], m_k_, P_k_)

            # Appending updated mean/covarince lists
            m_k_list .append(self.m_k)
            P_k_list .append(self.P_k)


        ## Performing Smoothing step of the Kalman Smoother
        
        # Reversing Filter lists
        m_k__list.reverse()
        P_k__list.reverse()
        m_k_list.reverse()
        P_k_list.reverse()

        # Initializing lists for Kalman Smoother
        m_k_s_list = []  # Initialization
        P_k_s_list = []  # Initialization
        G_k_list = []  # Initialization

        # Initializing with final State Mean and Covariance vector and matrix respectively
        m_k_s_list.append(m_k_list[0])
        P_k_s_list.append(P_k_list[0])

        # FOR LOOP: For each element in the y_k_list
        for ii in range(Data_Len-1):

            # Computing Kalman Smoother Gain
            G_k = np.dot(np.dot(P_k_list[ii+1], self.A.transpose()), np.linalg.inv(P_k__list[ii]))

            # Computing Smoothed state mean vector
            m_k_s = m_k_list[ii+1] + np.dot(G_k, (m_k_s_list[ii] - m_k__list[ii]))

            # Computing Smoothed state covariance matrix
            P_k_s = P_k_list[ii+1] + np.dot(np.dot(G_k, (P_k_s_list[ii] - P_k__list[ii])), G_k.transpose())

            # Appending Smoothed mean/covariance lists
            G_k_list.append(G_k)
            m_k_s_list.append(m_k_s)
            P_k_s_list.append(P_k_s)

        # Reversing Smoother lists
        G_k_list.reverse()
        m_k_s_list.reverse()
        P_k_s_list.reverse()

        # Return statement
        return G_k_list, m_k_s_list, P_k_s_list
    

class Extended_Kalman_Filter_Smoother:
    """Creates an object to handle Extended Kalman Filtering and Smoothing for Linear Time Invariant (LTI) System

    Attributes:
        A (numpy.array): LTI system matrix
        B (numpy.array): LTI input matrix
        C (numpy.array): LTI output matrix
        D (numpy.array): LTI feedforward matrix
        m_k (numpy.array): State mean vector
        P_k (numpy.array): State covariance matrix
        Q (numpy.array): Process error covariance matrix
        R (numpy.array): Measurement error covariance matrix
    """
    
    def __init__(self, f, F, h, H, m_k, P_k, Q, R):
        """Initializes the instance of class Extended_Kalman_Filter_Smoother

        Args:
            f (function(state,input) => numpy.array): Non-linear dynamics function which outputs state vector
            F (function(state,input) => numpy.array): Linearized system dynamics which outputs a matrix
            h (function(state) => numpy.array): Non-linear output function which outputs mesurement vector
            H (function(state) => numpy.array): Linearized output function which outputs a matrix
            m_k (numpy.array): State mean vector
            P_k (numpy.array): State covariance matrix
            Q (numpy.array): Process error covariance matrix
            R (numpy.array): Measurement error covariance matrix
        """

        # Initializing the instance
        self.f = f
        self.F = F
        self.h = h
        self.H = H
        self.m_k = m_k
        self.P_k = P_k
        self.Q = Q
        self.R = R

    def set_f(self, f):
        """Sets the system dynamics function for a non-linear time-varying (LTV) system
        
        Args:
            f (function(state,input) => numpy.array): Non-linear dynamics function which outputs state vector
        """
        self.f = f

    def set_F(self, F):
        """Sets the linearized system dynamics function for a non-linear time-varying (LTV) system
        
        Args:
            F (function(state,input) => numpy.array): Linearized system dynamics which outputs a matrix
        """
        self.F = F

    def set_h(self, h):
        """Sets the measurement function for a non-linear time-varying (LTV) system
        
        Args:
           h (function(state) => numpy.array): Non-linear output function which outputs mesurement vector
        """
        self.h = h

    def set_H(self, H):
        """Sets the linearized measurement function for a non-linear time-varying (LTV) system
        
        Args:
            H (function(state) => numpy.array): Linearized output function which outputs a matrix
        """
        self.H = H

    def set_Q(self, Q):
        """Sets the Q matrix for the system
        
        Args:
            Q (numpy.array): Process error covariance matrix
        """
        self.Q = Q

    def set_R(self, R):
        """Sets the R matrix for the system
        
        Args:
            R (numpy.array): Measurement error covariance matrix
        """
        self.R = R

    def Extended_Kalman_Predict(self, u_k):
        """Performs the predict step of the Extended Kalman Filter

        Args:
            u_k (numpy.array): Input to the system

        Returns:
            m_k_ (numpy.array): Predicted state mean vector
            P_k_ (numpy.array): Predicted state covariance matrix
        """

        # Computing predicted mean of the states
        m_k_ = self.f(self.m_k, u_k)

        # Computing predicted covariance of the states
        P_k_ = np.dot(np.dot(self.F(self.m_k, u_k), self.P_k), self.F(self.m_k, u_k).transpose()) + self.Q

        # Return statement
        return m_k_, P_k_

    def Extended_Kalman_Update(self, y_k, m_k_, P_k_):
        """Performs the update step of the Extended Kalman Filter

        Args:
            y_k (numpy.array): New measurement from the system
            m_k_ (numpy.array): Predicted state mean vector
            P_k_ (numpy.array): Predicted state covariance matrix

        Returns:
            v_k (numpy.array): Updated measurement error
            S_k (numpy.array): Updated measurement covariance matrix
            K_k (numpy.array): Updated Extended Kalman Gain
        """

        # Computing measurement error
        v_k = y_k - (self.h(m_k_))

        # Computing measurement covariance
        S_k = np.dot(np.dot(self.H(m_k_), P_k_), self.H(m_k_).transpose()) + self.R

        # Computing Kalman Gain
        K_k = np.dot(np.dot(P_k_, self.H(m_k_).transpose()), np.linalg.inv(S_k))

        # Updating state mean vector
        self.m_k = m_k_ + np.dot(K_k, v_k)

        # Updating state covariance matix
        self.P_k = P_k_ - np.dot(np.dot(K_k, S_k), K_k.transpose())

        # Return statement
        return v_k, S_k, K_k

    def Extended_Kalman_Smoother(self, u_k_list, y_k_list):
        """Performs the smoothening step of the Extended Kalman Smoother

        Args:
            u_k_list (list of numpy.array): List of inputs to the system
            y_k_list (list of numpy.array): List of measurements from the system

        Returns:
            G_k_list (list of numpy.array): List of Extended Kalman Smoother Gain
            m_k_s (list of numpy.array): List of smoothed state vectors
            P_k_s (list of numpy.array): List of smoothed state covarivance matrices
        """

        # Computing length of data
        Data_Len = len(y_k_list)

        ## Performing Filtering step of the Kalman Filter
        m_k__list = []  # Initializing
        P_k__list = []  # Initializing
        m_k_list = []  # Initializing
        P_k_list = []  # Initializing

        # Initializing with initial State Mean and Covariance vector and matrix respectively
        m_k_list.append(self.m_k)
        P_k_list.append(self.P_k)

        # FOR LOOP: For each element in the y_k_list
        for ii in range(Data_Len):

            # Computing predict step of Kalman Filter
            m_k_, P_k_ = self.Extended_Kalman_Predict(self, u_k_list[ii])

            # Appending predicted mean/covariance lists
            m_k__list .append(m_k_)
            P_k__list .append(P_k_)

            # Computing update step of Kalman Filter
            self.Extended_Kalman_Update(self, y_k_list[ii], m_k_, P_k_)

            # Appending updated mean/covarince lists
            m_k_list .append(self.m_k)
            P_k_list .append(self.P_k)


        ## Performing Smoothing step of the Kalman Smoother
        
        # Reversing Filter lists
        m_k__list.reverse()
        P_k__list.reverse()
        m_k_list.reverse()
        P_k_list.reverse()

        u_k_list.reverse()

        # Initializing lists for Kalman Smoother
        m_k_s_list = []  # Initialization
        P_k_s_list = []  # Initialization
        G_k_list = []  # Initialization

        # Initializing with final State Mean and Covariance vector and matrix respectively
        m_k_s_list.append(m_k_list[0])
        P_k_s_list.append(P_k_list[0])

        # FOR LOOP: For each element in the y_k_list
        for ii in range(Data_Len-1):

            # Computing Kalman Smoother Gain
            G_k = np.dot(np.dot(P_k_list[ii+1], self.F(m_k_list[ii+1], u_k_list[ii+1]).transpose()), np.linalg.inv(P_k__list[ii]))

            # Computing Smoothed state mean vector
            m_k_s = m_k_list[ii+1] + np.dot(G_k, (m_k_s_list[ii] - m_k__list[ii]))

            # Computing Smoothed state covariance matrix
            P_k_s = P_k_list[ii+1] + np.dot(np.dot(G_k, (P_k_s_list[ii] - P_k__list[ii])), G_k.transpose())

            # Appending Smoothed mean/covariance lists
            G_k_list.append(G_k)
            m_k_s_list.append(m_k_s)
            P_k_s_list.append(P_k_s)

        # Reversing Smoother lists
        G_k_list.reverse()
        m_k_s_list.reverse()
        P_k_s_list.reverse()

        # Return statement
        return G_k_list, m_k_s_list, P_k_s_list


class Unscented_Kalman_Filter_Smoother:
    """Creates an object to handle Unscented Kalman Filtering and Smoothing for Linear Time Invariant (LTI) System

    Attributes:
        f (function(state,input) => numpy.array): Non-linear dynamics function which outputs state vector
        h (function(state) => numpy.array): Non-linear output function which outputs mesurement vector
        n (integer): Number of states
        alpha (float): Parameter for the UKF algorithm
        k (float): Parameter for the UKF algorithm 
        beta (float): Parameter of the UKF algorithm       
        m_k (numpy.array): State mean vector
        P_k (numpy.array): State covariance matrix
        Q (numpy.array): PRocess error covariance matrix
        R (numpy.array): Measurement error covariance matrix
    """

    def __init__(self, f, h, n, m, alpha, k, beta, m_k, P_k, Q, R):
        """Initializes the instance of class Extended_Kalman_Filter_Smoother

        Args:
            f (function(state,input) => numpy.array): Non-linear dynamics function which outputs state vector
            h (function(state) => numpy.array): Non-linear output function which outputs mesurement vector
            n (integer): Number of states
            m (integer): Number of outputs
            alpha (float): Parameter for the UKF algorithm
            k (float): Parameter for the UKF algorithm        
            m_k (numpy.array): State mean vector
            P_k (numpy.array): State covariance matrix
            Q (numpy.array): PRocess error covariance matrix
            R (numpy.array): Measurement error covariance matrix
        """

        # Initializing the instance
        self.f = f
        self.h = h
        self.n = n
        self.m = m
        self.beta = beta        
        self.m_k = m_k
        self.P_k = P_k
        self.Q = Q
        self.R = R

        # Computing Lambda
        self.Lambda = (alpha**2)*(n+k)-n

        # Computing Wieghts for the UKF Algorithm
        self.W_0_m = (self.Lambda)/(n + self.Lambda)
        self.W_i_m = (1.0)/(2*(n + self.Lambda))
        self.W_0_c = (self.Lambda)/(n + self.Lambda) + (1 - alpha**2 + beta)
        self.W_i_c = (1.0)/(2*(n + self.Lambda))

    def set_f(self, f):
        """Sets the system dynamics function for a non-linear time-varying (LTV) system
        
        Args:
            f (function(state,input) => numpy.array): Non-linear dynamics function which outputs state vector
        """
        self.f = f

    def set_h(self, h):
        """Sets the measurement function for a non-linear time-varying (LTV) system
        
        Args:
           h (function(state) => numpy.array): Non-linear output function which outputs mesurement vector
        """
        self.h = h

    def set_Q(self, Q):
        """Sets the Q matrix for the system
        
        Args:
            Q (numpy.array): Process error covariance matrix
        """
        self.Q = Q

    def set_R(self, R):
        """Sets the R matrix for the system
        
        Args:
            R (numpy.array): Measurement error covariance matrix
        """
        self.R = R

    def __UKF_SigmPoints_Generator(self, m, P):
        """Computes Sigma Points for Unscented Kalman Filter

        Args:
            m (numpy.array): State mean vector
            P (np.array): State covariance matrix

        Returns:
            Sigma_X_Points_list (list of numpy.array): List of Sigma points
        """

        Sigma_X_Points_list = [] # Initialization

        # Computing squareroot of required elements
        sqrt_n_Lambda = mt.sqrt(self.n + self.Lambda)

        # Computing Eigenvalues and Left/Right Eigenvectors of P_k
        P_k_EigValues, P_k_Right_EigVec = np.linalg.eig(P)

        # Computing squareroot of the Diagonalized P_k matrix
        P_k_EigValues_Diag_Sqrt = np.sqrt(np.diag(P_k_EigValues))

        # Computing squareroot of P_k
        sqrt_P_k = np.abs(np.dot(P_k_Right_EigVec,np.dot(P_k_EigValues_Diag_Sqrt, np.linalg.inv(P_k_Right_EigVec))))

        # FOR LOOP: For 2n+1 Sigma Points
        for ii in range(2*self.n+1):

            if (ii == 0):

                # Computing Sigma Point
                Sigma_X_Point = m

                # Appending Sigma_X_Points_list
                Sigma_X_Points_list.append(Sigma_X_Point)

            elif ((ii > 0) and (ii <= self.n)):

                # Computing Sigma Point
                Sigma_X_Point = m + (sqrt_n_Lambda * sqrt_P_k[:,ii-1])

                # Appending Sigma_X_Points_list
                Sigma_X_Points_list.append(Sigma_X_Point)

            else:

                # Computing Sigma Point
                Sigma_X_Point = m - (sqrt_n_Lambda * sqrt_P_k[:,ii-(self.n+1)])

                # Appending Sigma_X_Points_list
                Sigma_X_Points_list.append(Sigma_X_Point)

            # Return statement
            return Sigma_X_Points_list
        
    def __UKF_SigmPoints_DynamicModel(self, Sigma_X_Points_list, u_k):
        """Computes Sigma Points through the dyanmic model for the Unscented Kalman Filter

        Args:
            Sigma_X_Points_list (list of numpy.array): List of Sigma points
            u_k (numpy.array): Input to the system

        Returns:
            Sigma_X_Points_Tilde_list  (list of numpy.array): List of Sigma points passed through the dynamic model
        """       
        
        Sigma_X_Points_Tilde_list = []  # Initialization

        # FOR LOOP: For each Sigma Point in Sigma_X_Points_list
        for ii in range(len(Sigma_X_Points_list)):

            # Propogating Sigma Point through Dynamic Model
            Sigma_X_Point_Tilde = self.f(Sigma_X_Points_list[ii], u_k)

            # Appending Sigma_X_Points_Tilde_list
            Sigma_X_Points_Tilde_list.append(Sigma_X_Point_Tilde)

        # Return statement
        return Sigma_X_Points_Tilde_list
    
    def __UKF_SigmPoints_MeasurementModel(self, Sigma_X_Points_list):
        """Computes Sigma Points through the measurement model for the Unscented Kalman Filter

        Args:
            Sigma_X_Points_list (list of numpy.array): List of Sigma points

        Returns:
            Sigma_Y_Points_Tilde_list  (list of numpy.array): List of Sigma points passed through the measurement model
        """       
        
        Sigma_Y_Points_Tilde_list = []  # Initialization

        # FOR LOOP: For each Sigma Point in Sigma_X_Points_list
        for ii in range(len(Sigma_X_Points_list)):

            # Propogating Sigma Point through Measurement Model
            Sigma_Y_Point_Tilde = self.h(Sigma_X_Points_list[ii])

            # Appending Sigma_X_Points_Tilde_list
            Sigma_Y_Points_Tilde_list.append(Sigma_Y_Point_Tilde)

        # Return statement
        return Sigma_Y_Points_Tilde_list

    def __UKF_Predict_State_MeanCovariance(self, Sigma_X_Points_list, Sigma_X_Points_Tilde_list):
        """Computes predict step state mean, covariance and cross-coavriance for the Unscented Kalman Filter

        Args:
            Sigma_X_Points_list (list of numpy.array): List of Sigma points
            Sigma_X_Points_Tilde_list  (list of numpy.array): List of Sigma points passed through the dynamic model

        Returns:
            m_k_ (numpy.array): Predicted state mean vector
            P_k_ (numpy.array): Predicted state covariance matrix
            D_k (numpy.array): Predicted state cross-covariance
        """    

        # Computing predicted mean of the state
        m_k_ = np.zeros((self.n, 1)) # Intialization

        # FOR LOOP: For each Sigma Point in Sigma_X_Points_Tilde_list
        for ii in range(len(Sigma_X_Points_Tilde_list)):

            if (ii == 0):

                m_k_ = m_k_ + (self.W_0_m * Sigma_X_Points_Tilde_list[ii]) 

            else:

                m_k_ = m_k_ + (self.W_i_m * Sigma_X_Points_Tilde_list[ii]) 

        # Computing predicted covariance of the state
        P_k_ = np.zeros((self.n, self.n)) # Intialization

        # FOR LOOP: For each Sigma Point in Sigma_X_Points_Tilde_list
        for ii in range(len(Sigma_X_Points_Tilde_list)):

            if (ii == 0):

                Sigma_m_k = Sigma_X_Points_Tilde_list[ii] - m_k_ 

                P_k_ = P_k_ + (self.W_0_c * np.dot(Sigma_m_k, Sigma_m_k.transpose())) 

            else:

                Sigma_m_k = Sigma_X_Points_Tilde_list[ii] - m_k_ 

                P_k_ = P_k_ + (self.W_i_c * np.dot(Sigma_m_k, Sigma_m_k.transpose()))  

        P_k_ = P_k_ + self.Q

        # Computing predicted cross-covariance of the state
        D_k = np.zeros((self.n, self.n)) # Intialization

        # FOR LOOP: For each Sigma Point in Sigma_X_Points_Tilde_list
        for ii in range(len(Sigma_X_Points_Tilde_list)):

            if (ii == 0):

                Sigma_m_k = Sigma_X_Points_list[ii] - self.m_k 

                Sigma_Tilde_m_k = Sigma_X_Points_Tilde_list[ii] - m_k_ 

                D_k = D_k + (self.W_0_c * np.dot(Sigma_m_k, Sigma_Tilde_m_k.transpose())) 

            else:

                Sigma_m_k = Sigma_X_Points_list[ii] - self.m_k 

                Sigma_Tilde_m_k = Sigma_X_Points_Tilde_list[ii] - m_k_ 

                D_k = D_k + (self.W_i_c * np.dot(Sigma_m_k, Sigma_Tilde_m_k.transpose()))

        # Return statement
        return m_k_, P_k_, D_k  
    
    def __UKF_Update_StateMeasurement_MeanCovariance(self, Sigma_X_Points_list, Sigma_Y_Points_Tilde_list, m_k_):
        """Computes predict step state mean and coavriance for the Unscented Kalman Filter

        Args:
            Sigma_X_Points_list (list of numpy.array): List of Sigma points
            Sigma_Y_Points_Tilde_list  (list of numpy.array): List of Sigma points passed through the measurement model
            m_k_ (numpy.array): Predicted state mean vector

        Returns:
            mu_k (numpy.array): Predicted measurement mean
            S_k (numpy.array): Predicted measurement covariance matrix
            C_k (numpy.array): Predicted state-measurement covariance matrix
        """    

        # Computing predicted mean of the measurement
        mu_k = np.zeros((self.m, 1)) # Intialization

        # FOR LOOP: For each Sigma Point in Sigma_Y_Points_Tilde_list
        for ii in range(len(Sigma_Y_Points_Tilde_list)):

            if (ii == 0):

                mu_k = mu_k + (self.W_0_m * Sigma_Y_Points_Tilde_list[ii]) 

            else:

                mu_k = mu_k + (self.W_i_m * Sigma_Y_Points_Tilde_list[ii]) 

        # Computing predicted covariance of the measurement
        S_k = np.zeros((self.m, self.m)) # Intialization

        # FOR LOOP: For each Sigma Point in Sigma_Y_Points_Tilde_list
        for ii in range(len(Sigma_Y_Points_Tilde_list)):

            if (ii == 0):

                Sigma_mu_k = Sigma_Y_Points_Tilde_list[ii] - mu_k 

                S_k = S_k + (self.W_0_c * np.dot(Sigma_mu_k, Sigma_mu_k.transpose())) 

            else:

                Sigma_mu_k = Sigma_Y_Points_Tilde_list[ii] - mu_k 

                S_k = S_k + (self.W_i_c * np.dot(Sigma_mu_k, Sigma_mu_k.transpose()))  

        S_k = S_k + self.R

        # Computing predicted cross-covariance between state and measurement
        C_k = np.zeros((self.n, self.m)) # Intialization

        # FOR LOOP: For each Sigma Point in Sigma_Y_Points_Tilde_list
        for ii in range(len(Sigma_Y_Points_Tilde_list)):

            if (ii == 0):

                Sigma_m_k_ = Sigma_X_Points_list[ii] - m_k_ 

                Sigma_mu_k = Sigma_Y_Points_Tilde_list[ii] - mu_k 

                C_k = C_k + (self.W_0_c * np.dot(Sigma_m_k_, Sigma_mu_k.transpose())) 

            else:

                Sigma_m_k_ = Sigma_X_Points_list[ii] - m_k_ 

                Sigma_mu_k = Sigma_Y_Points_Tilde_list[ii] - mu_k 

                C_k = C_k + (self.W_i_c * np.dot(Sigma_m_k_, Sigma_mu_k.transpose()))  

        # Return statement
        return mu_k, S_k, C_k

    def Unscented_Kalman_Predict(self, u_k):
        """Performs the predict step of the Unscented Kalman Filter

        Args:
            u_k (numpy.array): Input to the system

        Returns:
            m_k_ (numpy.array): Predicted state mean vector
            P_k_ (numpy.array): Predicted state covariance matrix            
            D_k (numpy.array): Predicted state cross-covariance
        """

        ## Forming Sigma Points
        Sigma_X_Points_list = self.__UKF_SigmPoints_Generator(self.m_k, self.P_k)

        ## Propagating Sigma Points through the Dynamic Model
        Sigma_X_Points_Tilde_list = self.__UKF_SigmPoints_DynamicModel(Sigma_X_Points_list, u_k)

        ## Computing predicted mean and covariance of the state
        m_k_, P_k_, D_k = self.__UKF_Predict_State_MeanCovariance(Sigma_X_Points_list, Sigma_X_Points_Tilde_list)

        # Return statement
        return m_k_, P_k_, D_k
    
    def Unscented_Kalman_Update(self, y_k, m_k_, P_k_):
        """Performs the update step of the Unscented Kalman Filter

        Args:
            y_k (numpy.array): New measurement from the system
            m_k_ (numpy.array): Predicted state mean vector
            P_k_ (numpy.array): Predicted state covariance matrix

        Returns:
            mu_k (numpy.array): Predicted measurement mean
            S_k (numpy.array): Predicted measurement covariance matrix
            C_k (numpy.array): Predicted state-measurement covariance matrix
            K_k (numpy.array): Updated Extended Kalman Gain
        """

        ## Forming Sigma Points
        Sigma_X_Points_list = self.__UKF_SigmPoints_Generator(m_k_, P_k_)

        ## Propagating Sigma Points through the Measurement Model
        Sigma_Y_Points_Tilde_list = self.__UKF_SigmPoints_MeasurementModel(Sigma_X_Points_list)

        ## Computing mean, covariance and cross-covariance between measurement and state
        mu_k, S_k, C_k = self.__UKF_Update_StateMeasurement_MeanCovariance(Sigma_X_Points_list, Sigma_Y_Points_Tilde_list, m_k_)

        ## Computing Filter Gain and Update state mean and covariance

        # Computing Kalman Gain
        K_k = np.dot(C_k, np.linalg.inv(S_k))

        # Updating state mean vector
        self.m_k = m_k_ + np.dot(K_k, (y_k - mu_k))

        # Updating state covariance matix
        self.P_k = P_k_ - np.dot(np.dot(K_k, S_k), K_k.transpose())

        # Return statement
        return mu_k, S_k, C_k, K_k
    
    def Unscented_Kalman_Smoother(self, u_k_list, y_k_list):
        """Performs the smoothening step of the Unscented Kalman Smoother

        Args:
            u_k_list (list of numpy.array): List of inputs to the system
            y_k_list (list of numpy.array): List of measurements from the system

        Returns:
            G_k_list (list of numpy.array): List of Extended Kalman Smoother Gain
            m_k_s (list of numpy.array): List of smoothed state vectors
            P_k_s (list of numpy.array): List of smoothed state covarivance matrices
        """

        # Computing length of data
        Data_Len = len(y_k_list)

        ## Performing Filtering step of the Kalman Filter
        m_k__list = []  # Initializing
        P_k__list = []  # Initializing
        D_k_list = []  # Initialization
        m_k_list = []  # Initializing
        P_k_list = []  # Initializing

        # Initializing with initial State Mean and Covariance vector and matrix respectively
        m_k_list.append(self.m_k)
        P_k_list.append(self.P_k)

        # FOR LOOP: For each element in the y_k_list
        for ii in range(Data_Len):

            # Computing predict step of Kalman Filter
            m_k_, P_k_, D_k = self.Unscented_Kalman_Predict(self, u_k_list[ii])

            # Appending predicted mean/covariance lists
            m_k__list .append(m_k_)
            P_k__list .append(P_k_)
            D_k_list .append(D_k)

            # Computing update step of Kalman Filter
            self.Unscented_Kalman_Update(self, y_k_list[ii], m_k_, P_k_)

            # Appending updated mean/covarince lists
            m_k_list .append(self.m_k)
            P_k_list .append(self.P_k)


        ## Performing Smoothing step of the Kalman Smoother
        
        # Reversing Filter lists
        m_k__list.reverse()
        P_k__list.reverse()
        D_k_list.reverse()
        m_k_list.reverse()
        P_k_list.reverse()

        u_k_list.reverse()

        # Initializing lists for Kalman Smoother
        m_k_s_list = []  # Initialization
        P_k_s_list = []  # Initialization
        G_k_list = []  # Initialization

        # Initializing with final State Mean and Covariance vector and matrix respectively
        m_k_s_list.append(m_k_list[0])
        P_k_s_list.append(P_k_list[0])

        # FOR LOOP: For each element in the y_k_list
        for ii in range(Data_Len-1):

            # Computing Kalman Smoother Gain
            G_k = np.dot(D_k_list[ii], np.linalg.inv(P_k__list[ii]))

            # Computing Smoothed state mean vector
            m_k_s = m_k_list[ii+1] + np.dot(G_k, (m_k_s_list[ii] - m_k__list[ii]))

            # Computing Smoothed state covariance matrix
            P_k_s = P_k_list[ii+1] + np.dot(np.dot(G_k, (P_k_s_list[ii] - P_k__list[ii])), G_k.transpose())

            # Appending Smoothed mean/covariance lists
            G_k_list.append(G_k)
            m_k_s_list.append(m_k_s)
            P_k_s_list.append(P_k_s)

        # Reversing Smoother lists
        G_k_list.reverse()
        m_k_s_list.reverse()
        P_k_s_list.reverse()

        # Return statement
        return G_k_list, m_k_s_list, P_k_s_list

        

