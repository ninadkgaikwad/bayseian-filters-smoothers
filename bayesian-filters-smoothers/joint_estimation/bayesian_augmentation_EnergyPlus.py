#-------------------------------------------------------------------Initial Setup------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#

## Importing external modules
import os
import sys
import numpy as np
import pandas as pd
import pickle
import datetime
import copy
import matplotlib.pyplot as plt
import matplotlib

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

#------------------------------------------------------------Getting True System Data--------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
# DateTime Tuple List for accessing desired data
DateTime_List_Tuple = [('02/01/2013', '02/14/2013')]

# Getting path for EnergyPlus Data 
EnergyPlus_Data_path = os.path.abspath(os.path.join('.','bayesian-filters-smoothers','data'))

# Printing the path for EnergyPlus Data 
print('EnergyPlus Data Path:' + '\n' + EnergyPlus_Data_path)

# Get Required Files from Sim_AggregatedTestTrainData_FolderPath
AggregatedTest_Dict_File = open(os.path.join(EnergyPlus_Data_path, 'Aggregation_DF_Test_Aggregation_Dict_1Zone_1.pickle'), "rb")
AggregatedTest_DF = pickle.load(AggregatedTest_Dict_File)
AggregatedTest_DF.reset_index(drop=True, inplace=True)

AggregatedTrain_Dict_File = open(os.path.join(EnergyPlus_Data_path, 'Aggregation_DF_Train_Aggregation_Dict_1Zone_1.pickle'), "rb")
AggregatedTrain_DF = pickle.load(AggregatedTrain_Dict_File)
AggregatedTrain_DF.reset_index(drop=True, inplace=True)

# Get Required Files from Sim_RegressionModelData_FolderPath
ANN_HeatInput_Test_DF_File = open(os.path.join(EnergyPlus_Data_path, 'ANN_HeatInput_Test_DF_1Zone_1.pickle'), "rb")
ANN_HeatInput_Test_DF = pickle.load(ANN_HeatInput_Test_DF_File)
ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)

ANN_HeatInput_Train_DF_File = open(os.path.join(EnergyPlus_Data_path, 'ANN_HeatInput_Train_DF_1Zone_1.pickle'), "rb")
ANN_HeatInput_Train_DF = pickle.load(ANN_HeatInput_Train_DF_File)
ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)

# Concatenating Train and Test Dataframes
Aggregated_DF = pd.concat([AggregatedTrain_DF,AggregatedTest_DF], axis=0)
Aggregated_DF.sort_values(by=["DateTime"])
Aggregated_DF.reset_index(drop=True, inplace=True)

ANN_HeatInput_DF = pd.concat([ANN_HeatInput_Train_DF,ANN_HeatInput_Test_DF], axis=0)
ANN_HeatInput_DF.sort_values(by=["DateTime"])
ANN_HeatInput_DF.reset_index(drop=True, inplace=True)

## Getting Desired Data based on Datetime Range

# Getting Start and End Dates for the Dataset
DateTime_List = Aggregated_DF['DateTime'].tolist()
StartDate_Dataset = DateTime_List[0]
EndDate_Dataset = DateTime_List[-1]

# Getting the File Resolution from DateTime_List
DateTime_Delta = DateTime_List[1] - DateTime_List[0]

FileResolution_Minutes = DateTime_Delta.seconds/60

# Initializing DateRange List
DateRange_Index = []

# FOR LOOP:
for DateTime_Tuple in DateTime_List_Tuple:

    # Getting Start and End Date
    StartDate = datetime.datetime.strptime(DateTime_Tuple[0],'%m/%d/%Y')
    EndDate = datetime.datetime.strptime(DateTime_Tuple[1],'%m/%d/%Y')

    # User Dates Corrected
    StartDate_Corrected = datetime.datetime(StartDate.year,StartDate.month,StartDate.day,0,int(FileResolution_Minutes),0)
    EndDate_Corrected = datetime.datetime(EndDate.year,EndDate.month,EndDate.day,23,60-int(FileResolution_Minutes),0)

    Counter_DateTime = -1

    # FOR LOOP:
    for Element in DateTime_List:
        Counter_DateTime = Counter_DateTime + 1

        if (Element >= StartDate_Corrected and Element <= EndDate_Corrected):
            DateRange_Index.append(Counter_DateTime)

# Getting Desired Dataset
Aggregated_DF_Desired = copy.deepcopy(Aggregated_DF.iloc[DateRange_Index,:])
Aggregated_DF_Desired.reset_index(drop=True, inplace=True)

ANN_HeatInput_DF_Desired = copy.deepcopy(ANN_HeatInput_DF.iloc[DateRange_Index,:])
ANN_HeatInput_DF_Desired.reset_index(drop=True, inplace=True)

DateTime_List_Desired = DateTime_List[DateRange_Index[0]:DateRange_Index[-1]+1]

## Getting Corrected ANN_HeatInput_DF_Desired

# Initialization
ANN_HeatInput_DF_Corrected = copy.deepcopy(pd.DataFrame())  

QZic = []
QZir = []
QSol1 = []
QSol2 = []
QAC = []

# FOR LOOP: Getting Summation
for ii in range(ANN_HeatInput_DF_Desired.shape[0]):
    QZic_1 = ANN_HeatInput_DF_Desired['QZic_P'][ii][0] + ANN_HeatInput_DF_Desired['QZic_L'][ii][0] + \
                    ANN_HeatInput_DF_Desired['QZic_EE'][ii][0]
    QZir_1 = ANN_HeatInput_DF_Desired['QZir_P'][ii][0] + ANN_HeatInput_DF_Desired['QZir_L'][ii][0] + \
                    ANN_HeatInput_DF_Desired['QZir_EE'][ii][0] + ANN_HeatInput_DF_Desired['QZivr_L'][ii][0]
    QZic.append(QZic_1)
    QZir.append(QZir_1)

    QSol1_1 = ANN_HeatInput_DF_Desired['QSol1'][ii][0]
    QSol2_1 = ANN_HeatInput_DF_Desired['QSol2'][ii][0]
    # QAC_Train_1 = ANN_HeatInput_DF_Desired['QAC'][ii][0]
    QAC_1 = AggregatedTrain_DF['QHVAC_X'].iloc[ii]

    QSol1.append(QSol1_1)
    QSol2.append(QSol2_1)
    QAC.append(QAC_1)

ANN_HeatInput_DF_Corrected.insert(0, 'QZic', QZic)
ANN_HeatInput_DF_Corrected.insert(0, 'QZir', QZir)
ANN_HeatInput_DF_Corrected.insert(0, 'QSol1', QSol1)
ANN_HeatInput_DF_Corrected.insert(0, 'QSol2', QSol2)
ANN_HeatInput_DF_Corrected.insert(0, 'QAC', QAC)
ANN_HeatInput_DF_Corrected.insert(0, 'DateTime', DateTime_List_Desired)

## Creating Measured Output and Measured Input for the System

# Creating Measured Output for the System - List of numpy.array
MeasuredOutput_y_List = []  # Initialization

# FOR LOOP: Getting Measured Output
for ii in range(Aggregated_DF_Desired.shape[0]):

    if (ii == 0): # Not considering first value as we dont have previous inputs for it

        continue    

    Zone_Air_Temperature = Aggregated_DF_Desired['Zone_Air_Temperature_'][ii]

    Zone_Air_Temperature = np.reshape(np.array([Zone_Air_Temperature]), (1,1))

    MeasuredOutput_y_List.append(Zone_Air_Temperature)

# Creating Measured Input for the System - List of numpy.array
MeasuredInput_u_List = []  # Initialization

# FOR LOOP: Getting Measured Output
for ii in range(ANN_HeatInput_DF_Corrected.shape[0]):

    if (ii == (ANN_HeatInput_DF_Corrected.shape[0]-1)): # Not considering the last value as we dont have a future measurement

        continue

    QZic = ANN_HeatInput_DF_Corrected['QZic'][ii]
    QZir = ANN_HeatInput_DF_Corrected['QZir'][ii]
    QSol1 = ANN_HeatInput_DF_Corrected['QSol1'][ii]
    QSol2 = ANN_HeatInput_DF_Corrected['QSol2'][ii]
    QAC = ANN_HeatInput_DF_Corrected['QAC'][ii]
    T_am = Aggregated_DF_Desired['Site_Outdoor_Air_Drybulb_Temperature_'][ii]

    Input = np.reshape(np.array([T_am, QAC, QZic, QZir, QSol1, QSol2]), (6,1))

    MeasuredInput_u_List.append(Input)

#--------------------------------------------------------------Defining System Model---------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
# Defining the Building Zone Discrete-Time Nonlinear System Function
def BuildingZone_f1(x_k_1, u_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        u_k_1 (numpy.array): Previous system measurement
        
    Returns:
        x_k_true (numpy.array): Current true system state
        x_k (numpy.array): Current noise corrupted system state       
    """
    
    # Simulation parameters
    ts = 300    
       
    # Getting individual state components
    Tz_k_1 = x_k_1[0,0]
    Tw_k_1 = x_k_1[1,0]

    # Getting the parameter components
    Rza_k_1 = x_k_1[2,0]
    Rzw_k_1 = x_k_1[3,0]
    Rwa_k_1 = x_k_1[4,0]
    Cz_k_1 = x_k_1[5,0]
    Cw_k_1 = x_k_1[6,0]
    Asol1_k_1 = x_k_1[7,0]
    Bsol1_k_1 = x_k_1[8,0]

    # Getting the input components
    T_am = u_k_1[0,0]
    QAC = u_k_1[1,0]
    QZic = u_k_1[2,0]
    QZir = u_k_1[3,0]
    QSol1 = u_k_1[4,0]
    QSol2 = u_k_1[5,0]
    
    # Computing current true system state
    Tz_k = Tz_k_1 + ts*((1/(Rzw_k_1*Cz_k_1))*(Tw_k_1 - Tz_k_1) + (1/(Rza_k_1*Cz_k_1))*(T_am - Tz_k_1) + (1/Cz_k_1)*(QZic - QAC) +(Asol1_k_1/Cz_k_1)*(QSol1))
    Tw_k = Tw_k_1 + ts*((1/(Rzw_k_1*Cw_k_1))*(Tz_k_1 - Tw_k_1) + (1/(Rwa_k_1*Cw_k_1))*(T_am - Tw_k_1) + (1/Cw_k_1)*(QZir) +(Bsol1_k_1/Cw_k_1)*(QSol2)) 

    # Computing current true system parameters
    Rza_k = Rza_k_1
    Rzw_k = Rzw_k_1
    Rwa_k = Rwa_k_1
    Cz_k = Cz_k_1
    Cw_k = Cw_k_1
    Asol1_k = Asol1_k_1
    Bsol1_k = Bsol1_k_1
                                        
    x_k_true = np.reshape(np.array([Tz_k, Tw_k, Rza_k, Rzw_k, Rwa_k, Cz_k, Cw_k, Asol1_k, Bsol1_k]),(9,1))
                                        
    # Return statement
    return x_k_true 

# Defining the Building Zone Discrete-Time Nonlinear Measurement Function
def BuildingZone_h1(x_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        
    Returns:
        y_k_true (numpy.array): Current true system measurement
        y_k (numpy.array): Current noise corrupted system measurement        
    """
    
    # Simulation parameters
    
    # Getting individual state components
    Tz_k_1 = x_k_1[0,0]
    
    # Computing current true system state
    Tz_k = Tz_k_1 
    
    # Computing current true system measurement
    y_k_true = Tz_k
    y_k_true = np.reshape(y_k_true,(1,1))
                                        
    # Return statement
    return y_k_true 


#----------------------------------------------------------Defining System Model Jacobian----------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
# Defining the Building Zone Discrete-Time Linearized System Function
def BuildingZone_F(x_k_1, u_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
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
    Rza_k_1 = x_k_1[2,0]
    Rzw_k_1 = x_k_1[3,0]
    Rwa_k_1 = x_k_1[4,0]
    Cz_k_1 = x_k_1[5,0]
    Cw_k_1 = x_k_1[6,0]
    Asol1_k_1 = x_k_1[7,0]
    Bsol1_k_1 = x_k_1[8,0]

    # Getting the input components
    T_am = u_k_1[0,0]
    QAC = u_k_1[1,0]
    QZic = u_k_1[2,0]
    QZir = u_k_1[3,0]
    QSol1 = u_k_1[4,0]
    QSol2 = u_k_1[5,0]
    
    # Computing System State Jacobian
    F = np.reshape(np.array([[-(1/Cz_k_1)*(1/Rza_k_1 + 1/Rzw_k_1), (1/(Cz_k_1*Rzw_k_1)), -(1/((Rza_k_1**2)*Cz_k_1))*(T_am - Tz_k_1) , -(1/((Rzw_k_1**2)*Cz_k_1))*(Tw_k_1 - Tz_k_1), 0, -(1/(Cz_k_1**2))*((1/(Rzw_k_1))*(Tw_k_1 - Tz_k_1) + (1/(Rza_k_1))*(T_am - Tz_k_1) + (QZic - QAC) +(Asol1_k_1)*(QSol1)), 0, (1/Cz_k_1)*(QSol1), 0], 
                             [(1/(Rzw_k_1*Cw_k_1)), -(1/Cw_k_1)*(1/Rwa_k_1 + 1/Rzw_k_1), 0, -(1/((Rzw_k_1**2)*Cw_k_1))*(Tz_k_1 - Tw_k_1), -(1/((Rwa_k_1**2)*Cw_k_1))*(T_am - Tw_k_1), 0, -(1/(Cw_k_1**2))*((1/(Rzw_k_1))*(Tz_k_1 - Tw_k_1) + (1/(Rwa_k_1))*(T_am - Tw_k_1) + (QZir) +(Bsol1_k_1)*(QSol2)), 0, (1/Cw_k_1)*(QSol2)], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0]]),(9,9))
    
                                        
    # Return statement
    return F 

# Defining the Building Zone Discrete-Time Linearized Measurement Function
def BuildingZone_H(x_k_1):
    """Provides discrete time dynamics for a nonlinear simple pendulum system
    
    Args:
        x_k_1 (numpy.array): Previous system state
        
    Returns:
        H (numpy.array): System linearized Measurement Jacobian      
    """
    
    # Computing System State Jacobian
    H = np.reshape(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),(1,9))    
                                        
    # Return statement
    return H 

#------------------------------------------------------------Plotting True System Data-------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#

## Plotting Setup

# Plot Size parameters
Plot_Width = 15
Plot_Height = 10

# Creating Figure
ax = plt.gca()

# Formatting X axis DateTime Ticks
# ax.xaxis.set_major_locator(matplotlib.dates.HourLocator((12,23)))
# ax.xaxis.set_major_locator(matplotlib.dates.DayLocator((1, 2, 3, 4, 5, 6, 7)))
# ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d/%Y'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%Y'))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d - %H:%M'))

## Simulation Plotting
# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting Measured output from the System - Tz
plt.subplot(111)
plt.plot(DateTime_List_Desired[1:], Aggregated_DF_Desired['Zone_Air_Temperature_'][1:], label=r'$T_{z}$ $(^{\circ}C)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Output '+ r'$(y)$', fontsize=12)
plt.title('Building Zone - Output Measurement', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.figure()

# Plotting Measured input from the System - T_am
plt.subplot(321)
plt.plot(DateTime_List_Desired[:-1], Aggregated_DF_Desired['Site_Outdoor_Air_Drybulb_Temperature_'][:-1], label=r'$T_{am}$ $(^{\circ}C)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Input '+ r'$(u)$', fontsize=12)
plt.title('Building Zone - Input Measurement', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting Measured input from the System - QAC
plt.subplot(322)
plt.plot(DateTime_List_Desired[:-1], ANN_HeatInput_DF_Corrected['QAC'][:-1], label=r'$Q_{ac}$ $(W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Input '+ r'$(u)$', fontsize=12)
plt.title('Building Zone - Input Measurement', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting Measured input from the System - QZic
plt.subplot(323)
plt.plot(DateTime_List_Desired[:-1], ANN_HeatInput_DF_Corrected['QZic'][:-1], label=r'$Q_{zic}$ $(W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Input '+ r'$(u)$', fontsize=12)
plt.title('Building Zone - Input Measurement', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting Measured input from the System - QZir
plt.subplot(324)
plt.plot(DateTime_List_Desired[:-1], ANN_HeatInput_DF_Corrected['QZir'][:-1], label=r'$Q_{zir}$ $(W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Input '+ r'$(u)$', fontsize=12)
plt.title('Building Zone - Input Measurement', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting Measured input from the System - QSol1
plt.subplot(325)
plt.plot(DateTime_List_Desired[:-1], ANN_HeatInput_DF_Corrected['QSol1'][:-1], label=r'$Q_{sol1}$ $(W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Input '+ r'$(u)$', fontsize=12)
plt.title('Building Zone - Input Measurement', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting Measured input from the System - QSol2
plt.subplot(326)
plt.plot(DateTime_List_Desired[:-1], ANN_HeatInput_DF_Corrected['QSol2'][:-1], label=r'$Q_{sol2}$ $(W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Input '+ r'$(u)$', fontsize=12)
plt.title('Building Zone - Input Measurement', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

#------------------------------------------------------------Simulation of Filter Step ------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
## System Model Setup

# Model input
u_model = MeasuredInput_u_List

## Filter Setup

# Initial Filter state mean/covariance (https://www.archtoolbox.com/r-values/ ; https://starcraftcustombuilders.com/insulation.htm)
Tz_ini_model = MeasuredOutput_y_List[0][0,0]  # Using Measurement of temperature to intialize
Tw_ini_model = MeasuredOutput_y_List[0][0,0]  # Using Measurement of temperature to intialize

Rza_ini_model = (5*(240./5.682)) + (5*(1.28/5.682))
Rzw_ini_model = (5*(240./5.682)) + (1/2)*(5*(1.28/5.682))
Rwa_ini_model = (1/2)*(5*(1.28/5.682))
Cz_ini_model = 10000000
Cw_ini_model = 10000000

Asol1_ini_model = 1.
Bsol1_ini_model = 1.

P_model = 1

# Filter constant parameters
n = 9  # Dimension of states of the system
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
u_k_model = u_model 

## Filter Computations

# Creating initial Filter mean vector
m_ini_model = np.reshape(np.array([Tz_ini_model, Tw_ini_model, Rza_ini_model, Rzw_ini_model, Rwa_ini_model, Cz_ini_model, Cw_ini_model, Asol1_ini_model, Bsol1_ini_model]), (9,1))

# Creating initial Filter state covariance matrix
P_ini_model = P_model*np.eye(9)

# Create the model Q and R matrices
Q_d_EKFS = np.reshape(np.array([[Q_model_EKFS, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, Q_model_EKFS, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, Q_params_EKFS, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, Q_params_EKFS, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, Q_params_EKFS, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, Q_params_EKFS, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, Q_params_EKFS, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, Q_params_EKFS, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, Q_params_EKFS]]), (9,9))
R_d_EKFS = np.reshape(np.array([R_model_EKFS]), (1,1))

Q_d_UKFS = np.reshape(np.array([[Q_model_UKFS, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, Q_model_UKFS, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, Q_params_UKFS, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, Q_params_UKFS, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, Q_params_UKFS, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, Q_params_UKFS, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, Q_params_UKFS, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, Q_params_UKFS, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, Q_params_UKFS]]), (9,9))
R_d_UKFS = np.reshape(np.array([R_model_UKFS]), (1,1))

Q_d_GFS = np.reshape(np.array([[Q_model_GFS, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, Q_model_GFS, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, Q_params_GFS, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, Q_params_GFS, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, Q_params_GFS, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, Q_params_GFS, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, Q_params_GFS, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, Q_params_GFS, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, Q_params_GFS]]), (9,9))
R_d_GFS = np.reshape(np.array([R_model_GFS]), (1,1))

## Creating Object for Noninear Model
BuildingZone_Nonlinear_EKF = EKFS(BuildingZone_f1, BuildingZone_F, BuildingZone_h1, BuildingZone_H, m_ini_model, P_ini_model, Q_d_EKFS, R_d_EKFS)
BuildingZone_Nonlinear_UKF = UKFS(BuildingZone_f1, BuildingZone_h1, n, m, alpha, k, beta, m_ini_model, P_ini_model, Q_d_UKFS, R_d_UKFS)
BuildingZone_Nonlinear_GF = GFS(GF_Type, BuildingZone_f1, BuildingZone_h1, n, m, m_ini_model, P_ini_model, Q_d_GFS, R_d_GFS, p)

## Filter Time Evolution
# Initializing model filter state array to store time evolution
x_model_nonlinear_filter_EKFS = m_ini_model

x_model_nonlinear_filter_UKFS = m_ini_model

x_model_nonlinear_filter_GFS = m_ini_model


# FOR LOOP: For each discrete time-step
for ii in range(len(MeasuredInput_u_List)):
    
    ## For measurements coming from Nonlinear System
    
    # Extended Kalman Filter: Predict Step    
    m_k_, P_k_ = BuildingZone_Nonlinear_EKF.Extended_Kalman_Predict(u_k_model[ii])
    
    # Extended Kalman Filter: Update Step
    v_k, S_k, K_k = BuildingZone_Nonlinear_EKF.Extended_Kalman_Update(MeasuredOutput_y_List[ii], m_k_, P_k_)    
    
    # Storing the Filtered states
    x_k_filter = BuildingZone_Nonlinear_EKF.m_k
    x_model_nonlinear_filter_EKFS = np.hstack((x_model_nonlinear_filter_EKFS, x_k_filter))


# FOR LOOP: For each discrete time-step
for ii in range(len(MeasuredInput_u_List)):
    
    ## For measurements coming from Nonlinear System
    
    # Extended Kalman Filter: Predict Step    
    m_k_, P_k_, D_k = BuildingZone_Nonlinear_UKF.Unscented_Kalman_Predict(u_k_model[ii])
    
    # Extended Kalman Filter: Update Step
    mu_k, S_k, C_k, K_k = BuildingZone_Nonlinear_UKF.Unscented_Kalman_Update(MeasuredOutput_y_List[ii], m_k_, P_k_)    
    
    # Storing the Filtered states
    x_k_filter = BuildingZone_Nonlinear_UKF.m_k
    x_model_nonlinear_filter_UKFS = np.hstack((x_model_nonlinear_filter_UKFS, x_k_filter))


# FOR LOOP: For each discrete time-step
for ii in range(len(MeasuredInput_u_List)):
    
    ## For measurements coming from Nonlinear System
    
    # Extended Kalman Filter: Predict Step    
    m_k_, P_k_, D_k = BuildingZone_Nonlinear_GF.Gaussian_Predict(u_k_model[ii])
    
    # Extended Kalman Filter: Update Step
    mu_k, S_k, C_k, K_k = BuildingZone_Nonlinear_GF.Gaussian_Update(MeasuredOutput_y_List[ii], m_k_, P_k_)    
    
    # Storing the Filtered states
    x_k_filter = BuildingZone_Nonlinear_GF.m_k
    x_model_nonlinear_filter_GFS = np.hstack((x_model_nonlinear_filter_GFS, x_k_filter))

## Filter Simulation Plotting

# Creating Figure
ax = plt.gca()

# Formatting X axis DateTime Ticks
# ax.xaxis.set_major_locator(matplotlib.dates.HourLocator((12,23)))
# ax.xaxis.set_major_locator(matplotlib.dates.DayLocator((1, 2, 3, 4, 5, 6, 7)))
# ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d/%Y'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%Y'))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d - %H:%M'))

# Setting Figure Size
plt.rcParams['figure.figsize'] = [Plot_Width, Plot_Height]

# Plotting Figures
plt.figure()

# Plotting True vs. Filtered States of Nonlinear System - Tz/Tw
plt.subplot(111)
plt.plot(DateTime_List_Desired[1:], Aggregated_DF_Desired['Zone_Air_Temperature_'][1:], label=r'$T_{z_{Measured}}$ $(^{\circ}C)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_EKFS[0:2,1:].transpose(), linestyle='--', linewidth=1, label=[r'$T_{z_{filter}}$ $(^{\circ}C)$', r'$T_{w_{filter}}$ $(^{\circ}C)$'])
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('State/Output '+ r'$(x/y)$', fontsize=12)
plt.title('Building Zone - Output Measurement EKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.figure()

# Plotting True vs. Filtered Parameters of Nonlinear System - Rza
plt.subplot(421)
plt.plot(DateTime_List_Desired[1:], Rza_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$R_{za_{ini}}$ $(^{\circ}C/W)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_EKFS[2,1:].transpose(), linestyle='--', linewidth=1, label=r'$R_{za_{filter}}$ $(^{\circ}C/W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter EKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Rzw
plt.subplot(422)
plt.plot(DateTime_List_Desired[1:], Rzw_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$R_{zw_{ini}}$ $(^{\circ}C/W)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_EKFS[3,1:].transpose(), linestyle='--', linewidth=1, label=r'$R_{zw_{filter}}$ $(^{\circ}C/W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter EKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Rwa
plt.subplot(423)
plt.plot(DateTime_List_Desired[1:], Rwa_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$R_{wa_{ini}}$ $(^{\circ}C/W)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_EKFS[4,1:].transpose(), linestyle='--', linewidth=1, label=r'$R_{wa_{filter}}$ $(^{\circ}C/W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter EKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Cz
plt.subplot(424)
plt.plot(DateTime_List_Desired[1:], Cz_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$C_{z_{ini}}$ $(J/^{\circ}C)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_EKFS[5,1:].transpose(), linestyle='--', linewidth=1, label=r'$C_{z_{filter}}$ $(J/^{\circ}C)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter EKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Cw
plt.subplot(425)
plt.plot(DateTime_List_Desired[1:], Cw_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$C_{w_{ini}}$ $(J/^{\circ}C)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_EKFS[6,1:].transpose(), linestyle='--', linewidth=1, label=r'$C_{w_{filter}}$ $(J/^{\circ}C)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter EKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Asol1
plt.subplot(426)
plt.plot(DateTime_List_Desired[1:], Asol1_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$A_{sol1_{ini}}$ ')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_EKFS[7,1:].transpose(), linestyle='--', linewidth=1, label=r'$A_{sol1_{filter}}$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter EKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Bsol1
plt.subplot(427)
plt.plot(DateTime_List_Desired[1:], Bsol1_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$B_{sol1_{ini}}$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_EKFS[8,1:].transpose(), linestyle='--', linewidth=1, label=r'$B_{sol1_{filter}}$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter EKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)



# Plotting Figures
plt.figure()

# Plotting True vs. Filtered States of Nonlinear System - Tz/Tw
plt.subplot(111)
plt.plot(DateTime_List_Desired[1:], Aggregated_DF_Desired['Zone_Air_Temperature_'][1:], label=r'$T_{z_{Measured}}$ $(^{\circ}C)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_UKFS[0:2,1:].transpose(), linestyle='--', linewidth=1, label=[r'$T_{z_{filter}}$ $(^{\circ}C)$', r'$T_{w_{filter}}$ $(^{\circ}C)$'])
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('State/Output '+ r'$(x/y)$', fontsize=12)
plt.title('Building Zone - Output Measurement UKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.figure()

# Plotting True vs. Filtered Parameters of Nonlinear System - Rza
plt.subplot(421)
plt.plot(DateTime_List_Desired[1:], Rza_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$R_{za_{ini}}$ $(^{\circ}C/W)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_UKFS[2,1:].transpose(), linestyle='--', linewidth=1, label=r'$R_{za_{filter}}$ $(^{\circ}C/W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter UKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Rzw
plt.subplot(422)
plt.plot(DateTime_List_Desired[1:], Rzw_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$R_{zw_{ini}}$ $(^{\circ}C/W)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_UKFS[3,1:].transpose(), linestyle='--', linewidth=1, label=r'$R_{zw_{filter}}$ $(^{\circ}C/W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter UKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Rwa
plt.subplot(423)
plt.plot(DateTime_List_Desired[1:], Rwa_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$R_{wa_{ini}}$ $(^{\circ}C/W)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_UKFS[4,1:].transpose(), linestyle='--', linewidth=1, label=r'$R_{wa_{filter}}$ $(^{\circ}C/W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter UKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Cz
plt.subplot(424)
plt.plot(DateTime_List_Desired[1:], Cz_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$C_{z_{ini}}$ $(J/^{\circ}C)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_UKFS[5,1:].transpose(), linestyle='--', linewidth=1, label=r'$C_{z_{filter}}$ $(J/^{\circ}C)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter UKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Cw
plt.subplot(425)
plt.plot(DateTime_List_Desired[1:], Cw_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$C_{w_{ini}}$ $(J/^{\circ}C)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_UKFS[6,1:].transpose(), linestyle='--', linewidth=1, label=r'$C_{w_{filter}}$ $(J/^{\circ}C)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter UKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Asol1
plt.subplot(426)
plt.plot(DateTime_List_Desired[1:], Asol1_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$A_{sol1_{ini}}$ ')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_UKFS[7,1:].transpose(), linestyle='--', linewidth=1, label=r'$A_{sol1_{filter}}$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter UKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Bsol1
plt.subplot(427)
plt.plot(DateTime_List_Desired[1:], Bsol1_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$B_{sol1_{ini}}$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_UKFS[8,1:].transpose(), linestyle='--', linewidth=1, label=r'$B_{sol1_{filter}}$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter UKFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)



# Plotting Figures
plt.figure()

# Plotting True vs. Filtered States of Nonlinear System - Tz/Tw
plt.subplot(111)
plt.plot(DateTime_List_Desired[1:], Aggregated_DF_Desired['Zone_Air_Temperature_'][1:], label=r'$T_{z_{Measured}}$ $(^{\circ}C)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_GFS[0:2,1:].transpose(), linestyle='--', linewidth=1, label=[r'$T_{z_{filter}}$ $(^{\circ}C)$', r'$T_{w_{filter}}$ $(^{\circ}C)$'])
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('State/Output '+ r'$(x/y)$', fontsize=12)
plt.title('Building Zone - Output Measurement GFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.figure()

# Plotting True vs. Filtered Parameters of Nonlinear System - Rza
plt.subplot(421)
plt.plot(DateTime_List_Desired[1:], Rza_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$R_{za_{ini}}$ $(^{\circ}C/W)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_GFS[2,1:].transpose(), linestyle='--', linewidth=1, label=r'$R_{za_{filter}}$ $(^{\circ}C/W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter GFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Rzw
plt.subplot(422)
plt.plot(DateTime_List_Desired[1:], Rzw_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$R_{zw_{ini}}$ $(^{\circ}C/W)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_GFS[3,1:].transpose(), linestyle='--', linewidth=1, label=r'$R_{zw_{filter}}$ $(^{\circ}C/W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter GFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Rwa
plt.subplot(423)
plt.plot(DateTime_List_Desired[1:], Rwa_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$R_{wa_{ini}}$ $(^{\circ}C/W)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_GFS[4,1:].transpose(), linestyle='--', linewidth=1, label=r'$R_{wa_{filter}}$ $(^{\circ}C/W)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter GFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Cz
plt.subplot(424)
plt.plot(DateTime_List_Desired[1:], Cz_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$C_{z_{ini}}$ $(J/^{\circ}C)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_GFS[5,1:].transpose(), linestyle='--', linewidth=1, label=r'$C_{z_{filter}}$ $(J/^{\circ}C)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter GFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Cw
plt.subplot(425)
plt.plot(DateTime_List_Desired[1:], Cw_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$C_{w_{ini}}$ $(J/^{\circ}C)$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_GFS[6,1:].transpose(), linestyle='--', linewidth=1, label=r'$C_{w_{filter}}$ $(J/^{\circ}C)$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter GFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Asol1
plt.subplot(426)
plt.plot(DateTime_List_Desired[1:], Asol1_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$A_{sol1_{ini}}$ ')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_GFS[7,1:].transpose(), linestyle='--', linewidth=1, label=r'$A_{sol1_{filter}}$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter GFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

# Plotting True vs. Filtered Parameters of Nonlinear System - Bsol1
plt.subplot(427)
plt.plot(DateTime_List_Desired[1:], Bsol1_ini_model*np.ones((len(DateTime_List_Desired[1:]),1)), label=r'$B_{sol1_{ini}}$')
plt.plot(DateTime_List_Desired[1:], x_model_nonlinear_filter_GFS[8,1:].transpose(), linestyle='--', linewidth=1, label=r'$B_{sol1_{filter}}$')
plt.xlabel('Time ' + r'$(mm/dd)$', fontsize=12)
plt.ylabel('Parameter '+ r'$(\theta)$', fontsize=12)
plt.title('Building Zone - Parameter GFS', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)

plt.show()

#-----------------------------------------------------------Simulation of Smoother Step -----------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------#
