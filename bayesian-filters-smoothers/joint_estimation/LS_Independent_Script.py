## Importing external modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from casadi import *
import pickle
import pandas as pd
import tensorflow as tf
import datetime
import math
import time
import tracemalloc
import copy
import scipy


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
 
## User Input: For the Filter : For ALL

Theta_Initial = [ 0.05, 0.05, 10000000, 10000000, 0.5, 0.5]  # [R_zw, R_wa, C_z, C_w, A_1, A_2]
 

# State/Parameter/Output/Input Size
n_State = 1  # Dimension of states of the system
n_Parameter = 5  # Dimension of parameters of the system
n_Output = 1  # Dimension of measurements of the system
n_Input = 6  # Dimension of inputs of the system

# Plot Size parameters
Plot_Width = 15
Plot_Height = 10

## Data Source dependent User Inputs
if (Type_Data_Source  == 1):  # PNNL Prototype Data

    Simulation_Name = "test1"

    Total_Aggregation_Zone_Number = 5

    ## Providing Parameter Details for storing Results based on Type of Model
    if (Type_System_Model == 1):  # 1-State

        State_Names = [r'$T_{z}$']

        Theta_Names = ['R_za' , 'C_z' , 'Asol']

    elif (Type_System_Model == 2):  # 2-State

        State_Names = [r'$T_{z}$', r'$T_{w}$']

        Theta_Names = ['R_zw', 'R_wa', 'C_z', 'C_w', 'A1', 'A2']

    ## User Input: Aggregation Unit Number ##
    # Aggregation_UnitNumber = 1
    # Aggregation_UnitNumber = 2

    # Aggregation Zone NameStem Input
    Aggregation_Zone_NameStem = 'Aggregation_Zone'

    

    ## Providing Proper Extensions depending on Type of Filter/Smoother Utilized and IMPROVEMENT is needed wrt type of data and model type (NINAD'S WORK)
    
    GBModel_Key = 'LS_DS_B' + '_SSM_' + str(Type_System_Model)

elif (Type_Data_Source  == 2):  # Simulated 4 State House Thermal Model Data

    Simulated_HouseThermalData_Filename = "PecanStreet_Austin_NSRDB_Gainesville_2017_Fall_3Months.mat"

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

    ## Providing Proper Extensions depending on Type of Filter/Smoother Utilized and IMPROVEMENT is needed wrt type of data and model type (NINAD'S WORK)
    GBModel_Key = 'LS_DS_H' + '_SSM_' + str(Type_System_Model)

## Basic Computation

# Computing ts in seconds
ts = FileResolution_Minutes*60

#training folder name
Training_FolderName = GBModel_Key

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
        Common_Data_DF = pd.concat([Aggregated_DF[[ 'DateTime', 'Zone_Air_Temperature_', 'Site_Outdoor_Air_Drybulb_Temperature_']], ANN_HeatInput_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']]], axis=1)
 
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

            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(np.array([Common_Data_DF_Test['Zone_Air_Temperature_'].iloc[0]]*n_State), (n_State,1))

            # Getting State Measurement
            y_measured = Common_Data_DF_Train['Zone_Air_Temperature_'].iloc[:]

            # Getting Disturbances
            T_a = Common_Data_DF_Train['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[:]
            Q_sol1 = Common_Data_DF_Train['QSol1_Corrected'].iloc[:]
            Q_sol2 = Common_Data_DF_Train['QSol2_Corrected'].iloc[:]
            Q_Zic = Common_Data_DF_Train['QZic'].iloc[:]
            Q_Zir = Common_Data_DF_Train['QZir'].iloc[:]
 
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

            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(np.array([Common_Data_DF_Test['Zone_Air_Temperature_'].iloc[0]]*n_State), (n_State,1))  

            # Getting State Measurement
            y_measured = Common_Data_DF_Train['Zone_Air_Temperature_'].iloc[:]

            # Getting Disturbances
            T_a = Common_Data_DF_Train['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[:]
            Q_sol1 = Common_Data_DF_Train['QSol1_Corrected'].iloc[:]
            Q_sol2 = Common_Data_DF_Train['QSol2_Corrected'].iloc[:]
            Q_Zic = Common_Data_DF_Train['QZic'].iloc[:]
            Q_Zir = Common_Data_DF_Train['QZir'].iloc[:]
   
 
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
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Test = Y_Measurement_Test[:,1:]
            # U_Measurement_Test = U_Measurement_Test[:,:-1]
                
            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(States_k_Array_Test[0,0:n_State], (n_State,1))  

            # Getting State Measurement
            y_measured= States_k_1_Array_Train[:,1]

            # Getting Disturbances
            T_sol_w = Inputs_k_Array_Train[:,1]
            T_sol_r = Inputs_k_Array_Train[:,2]
            T_am = Inputs_k_Array_Train[:,0]
            Q_in = Inputs_k_Array_Train[:,3]
            Q_ac = Inputs_k_Array_Train[:,4]
            Q_venti = Inputs_k_Array_Train[:,5]
            Q_infil = Inputs_k_Array_Train[:,6]
            Q_sol = Inputs_k_Array_Train[:,7]

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
                U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_sol_w, T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (6,1))))
 
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
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_sol_w, T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Test = Y_Measurement_Test[:,1:]
            # U_Measurement_Test = U_Measurement_Test[:,:-1]
                
            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(States_k_Array_Test[0,0:n_State], (n_State,1))  

            # Getting State Measurement
            y_measured= States_k_1_Array_Train[:,1]

            # Getting Disturbances
            T_sol_w = Inputs_k_Array_Train[:,1]
            T_sol_r = Inputs_k_Array_Train[:,2]
            T_am = Inputs_k_Array_Train[:,0]
            Q_in = Inputs_k_Array_Train[:,3]
            Q_ac = Inputs_k_Array_Train[:,4]
            Q_venti = Inputs_k_Array_Train[:,5]
            Q_infil = Inputs_k_Array_Train[:,6]
            Q_sol = Inputs_k_Array_Train[:,7]

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
                T_sol_r = Inputs_k_Array_Train[ii:,2]
                T_a = Inputs_k_Array_Train[ii,0]
                Q_int = Inputs_k_Array_Train[ii,3]
                Q_ac = Inputs_k_Array_Train[ii,4]
                Q_venti = Inputs_k_Array_Train[ii,5]
                Q_infil = Inputs_k_Array_Train[ii,6]
                Q_sol = Inputs_k_Array_Train[ii,7]
 
                # Updating Y_Measurement
                Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))
 
                # Updating U_Measurement
                U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_sol_w, T_sol_r, T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (6,1))))
 
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
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_sol_w, T_sol_r, T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Test = Y_Measurement_Test[:,1:]
            # U_Measurement_Test = U_Measurement_Test[:,:-1]  
                
            # Creating Initial Condition for Testing Simulations
            X_Initial_Test = np.reshape(States_k_Array_Test[0,0:n_State], (n_State,1))  

            # Getting State Measurement
            y_measured= States_k_1_Array_Train[:,1]

            # Getting Disturbances
            T_sol_w = Inputs_k_Array_Train[:,1]
            T_sol_r = Inputs_k_Array_Train[:,2]
            T_am = Inputs_k_Array_Train[:,0]
            Q_in = Inputs_k_Array_Train[:,3]
            Q_ac = Inputs_k_Array_Train[:,4]
            Q_venti = Inputs_k_Array_Train[:,5]
            Q_infil = Inputs_k_Array_Train[:,6]
            Q_sol = Inputs_k_Array_Train[:,7]

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
    # Estimation - Least Squares Method
    # =============================================================================
    
    # Timing Method
    Sim_StartTime = time.time()

    if (Type_Data_Source  == 1):  # PNNL Prototype Data
    
        if (Type_System_Model == 1):  # One State
    
            ## Initial Setup

            # State/Parameter/Output Dimensions
            State_n = n_State
            Parameter_n = n_Parameter
            Output_n = n_Output
            Input_n= n_Input

            # Initial Filter stae mean/covariance - as one state
            T_z_ini_model = y_measured #room temp

            y_ini_model = y_measured

            #parameters
            R_za_ini_model = Theta_Initial[0] ##-----------------------------------
            C_z_ini_model = Theta_Initial[1]  ##-----------------------------------
            Asol_ini_model = Theta_Initial[2]  ##-----------------------------------

            # Creating Infinity
            Infinity = np.inf


            ## Creating Optimization Variables

            # State Variables -------------------------------------------------------------------------------------------------------
            T_z = SX.sym('T_z',N+1,1)

            #Output Variable
            y_l = SX.sym('y_l',N,1)

            ## Getting total time steps
            N = y_measured.shape[0]

            # Parameter Variables
            R_za = SX.sym('R_za',1,1)
            C_z = SX.sym('C_z',1,1)
            Asol = SX.sym('Asol',1,1)


            # System Matrix
            A_matrix = SX.sym('A_matrix',State_n,State_n)

            A_matrix[0,0] = -1/(R_za*C_z)


            # System Constants
            C_matrix = DM(1,State_n)
            C_matrix[:,:] = np.hstack((np.array([[1]]),np.zeros((1,State_n-1))))  #np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

            #Creating input matrix, i.e B - [T_am[ii],Q_in[ii],Q_ac[ii],Q_sol[ii],Q_venti[ii],Q_infil[ii]]
            B_matrix = SX.sym('B_matrix',State_n,Input_n)

            B_matrix[0,0] = Asol/C_z
            B_matrix[0,1] = Asol/C_z
            B_matrix[0,2] = 1/C_z
            B_matrix[0,3] = 1/C_z
            B_matrix[0,4] = 1/C_z
          

            ## Constructing the Cost Function

            # Cost Function Development
            CostFunction = 0

            ## Constructing the Constraints

            # Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
            T_z_lb = []
            T_z_ub = []

            y_lb = []
            y_ub = []

            R_za_lb = [0]
            R_za_ub = [Infinity]
            C_z_lb = [0]
            C_z_ub = [Infinity]
            Asol_lb = [0]
            Asol_ub = [Infinity]




            Eq_x_lb = []
            Eq_y_lb = []

            Eq_x_ub = []
            Eq_y_ub = []

            Eq_x = []
            Eq_y = []

            # FOR LOOP: For each time step
            for ii in range(N):

                # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
                CostFunction += (y_l[ii]-y_measured[ii])**2

                ## State/Covariance Equations - Formulation

                # Creating State Vector
                x_k_1 = SX.sym('x_k_1',State_n,1)
                x_k = SX.sym('x_k',State_n,1)

                x_k_1[0,0] = T_z[ii+1]

                x_k[0,0] = T_z[ii]

                #Creating input vector - U

                U_vector = DM(Input_n,1)
                U_vector[:,:] = np.reshape(np.array([T_a[ii],Q_sol1[ii],Q_sol2[ii],Q_Zic[ii],Q_Zir[ii],Q_ac[ii]]), (Input_n,1))

                # State Equation
                x_Eq = -x_k_1 + x_k + ts*(A_matrix @ x_k + B_matrix @ U_vector)
                y_Eq = -y_l[ii] + C_matrix @ x_k #D matrix only for feed forward system


                # Adding current equations to Equation List
                Eq_x += [x_Eq] #[1,0] for 2 state
                Eq_y += [y_Eq] #always scalor for all states


                # Adding Equation Bounds, [0,0] for 2 equations
                Eq_x_lb += [0]
                Eq_x_ub += [0]

                Eq_y_lb += [0]
                Eq_y_ub += [0]


                # Adding Variable Bounds
                T_z_lb += [-Infinity]
                T_z_ub += [Infinity]

                y_lb += [-Infinity]
                y_ub += [Infinity]

            ## Adding Variable Bounds - For (N+1)th Variable
            T_z_lb += [-Infinity]
            T_z_ub += [Infinity]

            ## Constructing NLP Problem

            # Creating Optimization Variable: x
            x = vcat([R_za,C_z,Asol,T_z,y_l])

            # Creating Cost Function: J
            J = CostFunction

            # Creating Constraints: g
            g = vertcat(*Eq_x, *Eq_y)

            # Creating NLP Problem
            NLP_Problem = {'f': J, 'x': x, 'g': g}

            ## Constructiong NLP Solver
            NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

            ## Solving the NLP Problem

            # Creating Initial Variables
            T_z_l_ini = (np.vstack((T_z_ini_model, T_z_ini_model[-1]))).tolist()
            y_ini = y_ini_model.tolist()

            R_za_l_ini = R_za_ini_model*np.ones((1,)).tolist()
            C_z_l_ini = C_z_ini_model*np.ones((1,)).tolist()
            Asol_ini = Asol_ini_model*np.ones((1,)).tolist()

            

            x_initial = vertcat(*R_za_l_ini, *C_z_l_ini, *Asol_ini, *T_z_l_ini, *y_ini)

            # Creating Lower/Upper bounds on Variables and Equations
            x_lb = vertcat(*R_za_lb, *C_z_lb, *Asol_lb, *T_z_lb, *y_lb)

            x_ub = vertcat(*R_za_ub, *C_z_ub, *Asol_ub, *T_z_ub, *y_ub)

            G_lb = vertcat(*Eq_x_lb, *Eq_y_lb)

            G_ub = vertcat(*Eq_x_ub, *Eq_y_ub)

            # Solving NLP Problem
            NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

            #----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
            #--------------------------------------------------------------------------------------------------------------------------------------------------------#

            ## Getting the Solutions
            NLP_Sol = NLP_Solution['x'].full().flatten()

            R_za_sol = NLP_Sol[0]
            C_z_sol = NLP_Sol[1]
            Asol_sol = NLP_Sol[2]
            T_z_sol = NLP_Sol[3:(N+1)+3]
            y_sol = NLP_Sol[(N+1)+3:(N+1)+3+N]
    
        elif (Type_System_Model == 2): # Two State
    
            ## Initial Setup

            # State/Parameter/Output Dimensions
            State_n = n_State ##-----------------------------------
            Parameter_n = n_Parameter ##-----------------------------------
            Output_n = n_Output ##-----------------------------------
            Input_n = n_Input  ##-----------------------------------

            # Initial Filter stae mean/covariance - as one state
            T_z_ini_model = y_measured #room temp
            T_w_ini_model = y_measured

            y_ini_model = y_measured

            #parameters
            R_zw_ini_model = Theta_Initial[0]
            R_wa_ini_model = Theta_Initial[1]
            C_z_ini_model = Theta_Initial[2]
            C_w_ini_model = Theta_Initial[3]
            A1_ini_model = Theta_Initial[4]
            A2_ini_model = Theta_Initial[5]

            # Creating Infinity
            Infinity = np.inf


            ## Creating Optimization Variables

            # State Variables
            T_z = SX.sym('T_z',N+1,1)
            T_w = SX.sym('T_w',N+1,1)

            #Output Variable
            y_l = SX.sym('y_l',N,1)

            ## Getting total time steps
            N = y_measured.shape[0]

            # Parameter Variables
            R_zw = SX.sym('R_zw',1,1)
            R_wa = SX.sym('R_wa',1,1)
            C_z = SX.sym('C_z',1,1)
            C_w = SX.sym('C_w',1,1)
            A1 = SX.sym('A1',1,1)
            A2 = SX.sym('A2',1,1)
            
            

            # System Matrix
            A_matrix = SX.sym('A_matrix',State_n,State_n)

            A_matrix[0,0] = -(1/ (C_z*R_zw))
            A_matrix[0,1] = 1/(C_w*R_zw)

            A_matrix[1,0] = 1/(C_z*R_zw)
            A_matrix[1,1] = -(1/(C_w*R_zw))-(1/(C_w*R_wa))

            # System Constants
            C_matrix = DM(1,State_n)
            C_matrix[:,:] = np.hstack((np.array([[1]]),np.zeros((1,State_n-1)))) #np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

            #Creating input matrix, i.e B - [T_sol_w[ii], T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infil[ii]]
            B_matrix = SX.sym('B_matrix',State_n,Input_n)

            B_matrix[0,0] = 0
            B_matrix[0,1] = A1/C_z
            B_matrix[0,2] = 0
            B_matrix[0,3] = 1/C_z
            B_matrix[0,4] = 0
            B_matrix[0,5] = 1/C_z
            

            B_matrix[1,0] = 1/(C_w*R_wa)
            B_matrix[1,1] = 0
            B_matrix[1,2] = A2/C_w
            B_matrix[1,3] = 0
            B_matrix[1,4] = 1/C_w
            B_matrix[1,5] = 0


            ## Constructing the Cost Function

            # Cost Function Development
            CostFunction = 0

            ## Constructing the Constraints

            # Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
            T_z_lb = []
            T_z_ub = []

            T_w_lb = []
            T_w_ub = []

            y_lb = []
            y_ub = []

            R_zw_lb = [0]
            R_zw_ub = [Infinity]
            R_wa_lb = [0]
            R_wa_ub = [Infinity]
            C_z_lb = [0]
            C_z_ub = [Infinity]
            C_w_lb = [0]
            C_w_ub = [Infinity]
            A1_lb = [0]
            A1_ub = [Infinity]
            A2_lb = [0]
            A2_ub = [Infinity]



            Eq_x_lb = []
            Eq_y_lb = []

            Eq_x_ub = []
            Eq_y_ub = []

            Eq_x = []
            Eq_y = []

            # FOR LOOP: For each time step
            for ii in range(N):

                # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
                CostFunction += (y_l[ii]-y_measured[ii])**2

                ## State/Covariance Equations - Formulation

                # Creating State Vector
                x_k_1 = SX.sym('x_k_1',State_n,1)
                x_k = SX.sym('x_k',State_n,1)

                x_k_1[0,0] = T_z[ii+1]
                x_k_1[1,0] = T_w[ii+1]

                x_k[0,0] = T_z[ii]
                x_k[1,0] = T_w[ii]

                #Creating input vector - U

                U_vector = DM(Input_n,1)
                U_vector[:,:] = np.reshape(np.array([T_a[ii], Q_sol1[ii], Q_sol2[ii], Q_Zic[ii], Q_Zir[ii], Q_ac[ii]]), (Input_n,1))

                # State Equation
                x_Eq = -x_k_1 + x_k + ts*(A_matrix @ x_k + B_matrix @ U_vector)
                y_Eq = -y_l[ii] + C_matrix @ x_k #D matrix only for feed forward system


                # Adding current equations to Equation List
                Eq_x += [x_Eq[0,0], x_Eq[1,0]] #[1,0] for 2 state
                Eq_y += [y_Eq] #always scalor for all states


                # Adding Equation Bounds, [0,0] for 2 equations
                Eq_x_lb += [0,0]
                Eq_x_ub += [0,0]

                Eq_y_lb += [0]
                Eq_y_ub += [0]


                # Adding Variable Bounds
                T_z_lb += [-Infinity]
                T_z_ub += [Infinity]

                T_w_lb += [-Infinity]
                T_w_ub += [Infinity]

                y_lb += [-Infinity]
                y_ub += [Infinity]

            ## Adding Variable Bounds - For (N+1)th Variable
            T_z_lb += [-Infinity]
            T_z_ub += [Infinity]

            T_w_lb += [-Infinity]
            T_w_ub += [Infinity]

            ## Constructing NLP Problem

            # Creating Optimization Variable: x
            x = vcat([R_zw,R_wa,C_z,C_w,A1,A2,T_z,T_w,y_l])

            # Creating Cost Function: J
            J = CostFunction

            # Creating Constraints: g
            g = vertcat(*Eq_x, *Eq_y)

            # Creating NLP Problem
            NLP_Problem = {'f': J, 'x': x, 'g': g}

            ## Constructiong NLP Solver
            NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

            ## Solving the NLP Problem

            # Creating Initial Variables
            T_z_l_ini = (np.vstack((T_z_ini_model, T_z_ini_model[-1]))).tolist()
            T_w_l_ini = (np.vstack((T_w_ini_model, T_w_ini_model[-1]))).tolist()

            y_ini = y_measured.tolist()

            R_zw_l_ini = R_zw_ini_model*np.ones((1,)).tolist()
            R_wa_l_ini = R_wa_ini_model*np.ones((1,)).tolist()
            C_z_l_ini = C_z_ini_model*np.ones((1,)).tolist()
            C_w_l_ini = C_w_ini_model*np.ones((1,)).tolist()
            A1_ini = A1_ini_model*np.ones((1,)).tolist()
            A2_ini = A2_ini_model*np.ones((1,)).tolist()
           
            
            

            x_initial = vertcat(*R_zw_l_ini, *R_wa_l_ini, *C_z_l_ini, *C_w_l_ini, *A1_ini, *A2_ini, *T_z_l_ini, *T_w_l_ini, *y_ini)

            # Creating Lower/Upper bounds on Variables and Equations
            x_lb = vertcat(*R_zw_lb, *R_wa_lb, *C_z_lb, *C_w_lb, *A1_lb, *A2_lb, *T_z_lb, *T_w_lb, *y_lb)

            x_ub = vertcat(*R_zw_ub, *R_wa_ub, *C_z_ub, *C_w_ub, *A1_ub, *A2_ub, *T_z_ub, *T_w_ub, *y_ub)

            G_lb = vertcat(*Eq_x_lb, *Eq_y_lb)

            G_ub = vertcat(*Eq_x_ub, *Eq_y_ub)

            # Solving NLP Problem
            NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

            #----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
            #--------------------------------------------------------------------------------------------------------------------------------------------------------#

            ## Getting the Solutions
            NLP_Sol = NLP_Solution['x'].full().flatten()

            R_zw_sol = NLP_Sol[0]
            R_wa_sol = NLP_Sol[1]
            C_z_sol = NLP_Sol[2]
            C_w_sol = NLP_Sol[3]
            A1_sol = NLP_Sol[4]
            A2_sol = NLP_Sol[5]
            
            T_z_sol = NLP_Sol[6:(N+1)+6]
            T_w_sol = NLP_Sol[(N+1)+6:2*(N+1)+6]
            y_sol = NLP_Sol[2*(N+1)+6:2*(N+1)+6+N]
    
    elif (Type_Data_Source  == 2): # 4th Order ODE House Thermal Model
    
        if (Type_System_Model == 1):  # One State
    
            ## Initial Setup

            # State/Parameter/Output Dimensions
            State_n = n_State
            Parameter_n = n_Parameter
            Output_n = n_Output
            Input_n= n_Input

            # Initial Filter stae mean/covariance - as one state
            T_ave_ini_model = y_measured #room temp

            y_ini_model = y_measured

            #parameters
            R_win_ini_model = Theta_Initial[0] ##-----------------------------------
            C_in_ini_model = Theta_Initial[1]  ##-----------------------------------
            C1_ini_model = Theta_Initial[2]  ##-----------------------------------
            C2_ini_model = Theta_Initial[3]  ##-----------------------------------
            C3_ini_model = Theta_Initial[4]  ##-----------------------------------

            # Creating Infinity
            Infinity = np.inf


            ## Creating Optimization Variables

            # State Variables -------------------------------------------------------------------------------------------------------
            T_ave = SX.sym('T_ave',N+1,1)

            #Output Variable
            y_l = SX.sym('y_l',N,1)

            ## Getting total time steps
            N = y_measured.shape[0]

            # Parameter Variables
            R_win = SX.sym('R_win',1,1)
            C_in = SX.sym('C_in',1,1)
            C_1 = SX.sym('C_1',1,1)
            C_2 = SX.sym('C_2',1,1)
            C_3 = SX.sym('C_3',1,1)

            # System Matrix
            A_matrix = SX.sym('A_matrix',State_n,State_n)

            A_matrix[0,0] = -1/(R_win*C_in)


            # System Constants
            C_matrix = DM(1,State_n)
            C_matrix[:,:] = np.hstack((np.array([[1]]),np.zeros((1,State_n-1))))  #np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

            #Creating input matrix, i.e B - [T_am[ii],Q_in[ii],Q_ac[ii],Q_sol[ii],Q_venti[ii],Q_infil[ii]]
            B_matrix = SX.sym('B_matrix',State_n,Input_n)

            B_matrix[0,0] = 1/(R_win*C_in)
            B_matrix[0,1] = C_1/C_in
            B_matrix[0,2] = C_2/C_in
            B_matrix[0,3] = C_3/C_in
            B_matrix[0,4] = 1/C_in
            B_matrix[0,5] = 1/C_in

            ## Constructing the Cost Function

            # Cost Function Development
            CostFunction = 0

            ## Constructing the Constraints

            # Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
            T_ave_lb = []
            T_ave_ub = []

            y_lb = []
            y_ub = []

            R_win_lb = [0]
            R_win_ub = [Infinity]
            C_in_lb = [0]
            C_in_ub = [Infinity]
            C_1_lb = [0]
            C_1_ub = [Infinity]
            C_2_lb = [0]
            C_2_ub = [Infinity]
            C_3_lb = [0]
            C_3_ub = [Infinity]




            Eq_x_lb = []
            Eq_y_lb = []

            Eq_x_ub = []
            Eq_y_ub = []

            Eq_x = []
            Eq_y = []

            # FOR LOOP: For each time step
            for ii in range(N):

                # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
                CostFunction += (y_l[ii]-y_measured[ii])**2

                ## State/Covariance Equations - Formulation

                # Creating State Vector
                x_k_1 = SX.sym('x_k_1',State_n,1)
                x_k = SX.sym('x_k',State_n,1)

                x_k_1[0,0] = T_ave[ii+1]

                x_k[0,0] = T_ave[ii]

                #Creating input vector - U

                U_vector = DM(Input_n,1)
                U_vector[:,:] = np.reshape(np.array([T_am[ii],Q_in[ii],Q_ac[ii],Q_sol[ii],Q_venti[ii],Q_infil[ii]]), (Input_n,1))

                # State Equation
                x_Eq = -x_k_1 + x_k + ts*(A_matrix @ x_k + B_matrix @ U_vector)
                y_Eq = -y_l[ii] + C_matrix @ x_k #D matrix only for feed forward system


                # Adding current equations to Equation List
                Eq_x += [x_Eq[0,0]] #[1,0] for 2 state
                Eq_y += [y_Eq] #always scalor for all states


                # Adding Equation Bounds, [0,0] for 2 equations
                Eq_x_lb += [0]
                Eq_x_ub += [0]

                Eq_y_lb += [0]
                Eq_y_ub += [0]


                # Adding Variable Bounds
                T_ave_lb += [-Infinity]
                T_ave_ub += [Infinity]

                y_lb += [-Infinity]
                y_ub += [Infinity]

            ## Adding Variable Bounds - For (N+1)th Variable
            T_ave_lb += [-Infinity]
            T_ave_ub += [Infinity]

            ## Constructing NLP Problem

            # Creating Optimization Variable: x
            x = vcat([R_win,C_in,C_1,C_2,C_3,T_ave,y_l])

            # Creating Cost Function: J
            J = CostFunction

            # Creating Constraints: g
            g = vertcat(*Eq_x, *Eq_y)

            # Creating NLP Problem
            NLP_Problem = {'f': J, 'x': x, 'g': g}

            ## Constructiong NLP Solver
            NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

            ## Solving the NLP Problem

            # Creating Initial Variables
            T_ave_ini = (np.vstack((T_ave_ini_model, T_ave_ini_model[-1]))).tolist()
            y_ini = y_ini_model.tolist()

            R_win_ini = R_win_ini_model*np.ones((1,)).tolist()
            C_in_ini = C_in_ini_model*np.ones((1,)).tolist()
            C_1_ini = C_1_ini_model*np.ones((1,)).tolist()
            C_2_ini = C_2_ini_model*np.ones((1,)).tolist()
            C_3_ini = C_3_ini_model*np.ones((1,)).tolist()
            

            x_initial = vertcat(*R_win_ini, *C_in_ini, *C_1_ini, *C_2_ini, *C_3_ini, *T_ave_ini, *y_ini)

            # Creating Lower/Upper bounds on Variables and Equations
            x_lb = vertcat(*R_win_lb, *C_in_lb, *C_1_lb, *C_2_lb, *C_3_lb, *T_ave_lb, *y_lb)

            x_ub = vertcat(*R_win_ub, *C_in_ub, *C_1_ub, *C_2_ub, *C_3_ub, *T_ave_ub, *y_ub)

            G_lb = vertcat(*Eq_x_lb, *Eq_y_lb)

            G_ub = vertcat(*Eq_x_ub, *Eq_y_ub)

            # Solving NLP Problem
            NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

            #----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
            #--------------------------------------------------------------------------------------------------------------------------------------------------------#

            ## Getting the Solutions
            NLP_Sol = NLP_Solution['x'].full().flatten()

            R_win_sol = NLP_Sol[0]
            C_in_sol = NLP_Sol[1]
            C_1_sol = NLP_Sol[2]
            C_2_sol = NLP_Sol[3]
            C_3_sol = NLP_Sol[4]
            T_ave_sol = NLP_Sol[5:(N+1)+5]
            y_sol = NLP_Sol[(N+1)+5:(N+1)+5+N]
    
        elif (Type_System_Model == 2): # Two State
    
            ## Initial Setup

            # State/Parameter/Output Dimensions
            State_n = n_State ##-----------------------------------
            Parameter_n = n_Parameter ##-----------------------------------
            Output_n = n_Output ##-----------------------------------
            Input_n = n_Input  ##-----------------------------------

            # Initial Filter stae mean/covariance - as one state
            T_ave_ini_model = y_measured #room temp
            T_wall_ini_model = y_measured

            y_ini_model = y_measured

            #parameters
            R_win_ini_model = Theta_Initial[0]
            R_w2_ini_model = Theta_Initial[1]
            C_in_ini_model = Theta_Initial[2]
            C_w_ini_model = Theta_Initial[3]
            C_1_ini_model = Theta_Initial[4]
            C_2_ini_model = Theta_Initial[5]
            C_3_ini_model = Theta_Initial[6]
            

            # Creating Infinity
            Infinity = np.inf


            ## Creating Optimization Variables

            # State Variables
            T_ave = SX.sym('T_ave',N+1,1)
            T_wall = SX.sym('T_ave',N+1,1)

            #Output Variable
            y_l = SX.sym('y_l',N,1)

            ## Getting total time steps
            N = y_measured.shape[0]

            # Parameter Variables
            R_win = SX.sym('R_win',1,1)
            R_w2 = SX.sym('R_w2',1,1)
            C_in = SX.sym('C_in',1,1)
            C_w = SX.sym('C_w',1,1)
            C_1 = SX.sym('C_1',1,1)
            C_2 = SX.sym('C_2',1,1)
            C_3 = SX.sym('C_3',1,1)
            

            # System Matrix
            A_matrix = SX.sym('A_matrix',State_n,State_n)

            A_matrix[0,0] = -((1/((R_w2/2)*C_in)) + (1/(R_win*C_in)))
            A_matrix[0,1] = 1/((R_w2/2)*C_in)

            A_matrix[1,0] = 1/((R_w2/2)*C_w)
            A_matrix[1,1] = -2/((R_w2/2)*C_w)

            # System Constants
            C_matrix = DM(1,State_n)
            C_matrix[:,:] = np.hstack((np.array([[1]]),np.zeros((1,State_n-1)))) #np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

            #Creating input matrix, i.e B - [T_sol_w[ii], T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infil[ii]]
            B_matrix = SX.sym('B_matrix',State_n,Input_n)

            B_matrix[0,0] = 0
            B_matrix[0,1] = 1/(R_win*C_in)
            B_matrix[0,2] = C_1/C_in
            B_matrix[0,3] = C_2/C_in
            B_matrix[0,4] = 0
            B_matrix[0,5] = 1/C_in
            B_matrix[0,6] = 1/C_in

            B_matrix[1,0] = 1/((R_w2/2)*C_w)
            B_matrix[1,1] = 0
            B_matrix[1,2] = 0
            B_matrix[1,3] = 0
            B_matrix[1,4] = C_3/C_w
            B_matrix[1,5] = 0
            B_matrix[1,6] = 0

            ## Constructing the Cost Function

            # Cost Function Development
            CostFunction = 0

            ## Constructing the Constraints

            # Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
            T_ave_lb = []
            T_ave_ub = []

            T_wall_lb = []
            T_wall_ub = []

            y_lb = []
            y_ub = []

            R_win_lb = [0]
            R_win_ub = [Infinity]
            R_w2_lb = [0]
            R_w2_ub = [Infinity]
            C_in_lb = [0]
            C_in_ub = [Infinity]
            C_w_lb = [0]
            C_w_ub = [Infinity]
            C_1_lb = [0]
            C_1_ub = [Infinity]
            C_2_lb = [0]
            C_2_ub = [Infinity]
            C_3_lb = [0]
            C_3_ub = [Infinity]
            


            Eq_x_lb = []
            Eq_y_lb = []

            Eq_x_ub = []
            Eq_y_ub = []

            Eq_x = []
            Eq_y = []

            # FOR LOOP: For each time step
            for ii in range(N):

                # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
                CostFunction += (y_l[ii]-y_measured[ii])**2

                ## State/Covariance Equations - Formulation

                # Creating State Vector
                x_k_1 = SX.sym('x_k_1',State_n,1)
                x_k = SX.sym('x_k',State_n,1)

                x_k_1[0,0] = T_ave[ii+1]
                x_k_1[1,0] = T_wall[ii+1]

                x_k[0,0] = T_ave[ii]
                x_k[1,0] = T_wall[ii]

                #Creating input vector - U

                U_vector = DM(Input_n,1)
                U_vector[:,:] = np.reshape(np.array([T_sol_w[ii], T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infil[ii]]), (Input_n,1))

                # State Equation
                x_Eq = -x_k_1 + x_k + ts*(A_matrix @ x_k + B_matrix @ U_vector)
                y_Eq = -y_l[ii] + C_matrix @ x_k #D matrix only for feed forward system


                # Adding current equations to Equation List
                Eq_x += [x_Eq[0,0], x_Eq[1,0]] #[1,0] for 2 state
                Eq_y += [y_Eq] #always scalor for all states


                # Adding Equation Bounds, [0,0] for 2 equations
                Eq_x_lb += [0,0]
                Eq_x_ub += [0,0]

                Eq_y_lb += [0]
                Eq_y_ub += [0]


                # Adding Variable Bounds
                T_ave_lb += [-Infinity]
                T_ave_ub += [Infinity]

                T_wall_lb += [-Infinity]
                T_wall_ub += [Infinity]

                y_lb += [-Infinity]
                y_ub += [Infinity]

            ## Adding Variable Bounds - For (N+1)th Variable
            T_ave_lb += [-Infinity]
            T_ave_ub += [Infinity]

            T_wall_lb += [-Infinity]
            T_wall_ub += [Infinity]

            ## Constructing NLP Problem

            # Creating Optimization Variable: x
            x = vcat([R_win,R_w2,C_in,C_w,C_1,C_2,C_3,T_ave,T_wall,y_l])

            # Creating Cost Function: J
            J = CostFunction

            # Creating Constraints: g
            g = vertcat(*Eq_x, *Eq_y)

            # Creating NLP Problem
            NLP_Problem = {'f': J, 'x': x, 'g': g}

            ## Constructiong NLP Solver
            NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

            ## Solving the NLP Problem

            # Creating Initial Variables
            T_ave_ini = (np.vstack((T_ave_ini_model, T_ave_ini_model[-1]))).tolist()
            T_wall_ini = (np.vstack((T_wall_ini_model, T_wall_ini_model[-1]))).tolist()

            y_ini = y_measured.tolist()

            R_win_ini = R_win_ini_model*np.ones((1,)).tolist()
            R_w2_ini = R_win_ini_model*np.ones((1,)).tolist()
            C_in_ini = C_in_ini_model*np.ones((1,)).tolist()
            C_w_ini = C_3_ini_model*np.ones((1,)).tolist()
            C_1_ini = C_1_ini_model*np.ones((1,)).tolist()
            C_2_ini = C_2_ini_model*np.ones((1,)).tolist()
            C_3_ini = C_3_ini_model*np.ones((1,)).tolist()
            
            

            x_initial = vertcat(*R_win_ini, *R_w2_ini, *C_in_ini, *C_w_ini, *C_1_ini, *C_2_ini, *C_3_ini,  *T_ave_ini, *T_wall_ini, *y_ini)

            # Creating Lower/Upper bounds on Variables and Equations
            x_lb = vertcat(*R_win_lb, *R_w2_lb, *C_in_lb, *C_w_lb, *C_1_lb, *C_2_lb, *C_3_lb,  *T_ave_lb, *T_wall_lb, *y_lb)

            x_ub = vertcat(*R_win_ub, *R_w2_ub, *C_in_ub, *C_w_ub, *C_1_ub, *C_2_ub, *C_3_ub,  *T_ave_ub, *T_wall_ub, *y_ub)

            G_lb = vertcat(*Eq_x_lb, *Eq_y_lb)

            G_ub = vertcat(*Eq_x_ub, *Eq_y_ub)

            # Solving NLP Problem
            NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

            #----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
            #--------------------------------------------------------------------------------------------------------------------------------------------------------#

            ## Getting the Solutions
            NLP_Sol = NLP_Solution['x'].full().flatten()

            R_win_sol = NLP_Sol[0]
            R_w2_sol = NLP_Sol[1]
            C_in_sol = NLP_Sol[2]
            C_w_sol = NLP_Sol[3]
            C_1_sol = NLP_Sol[4]
            C_2_sol = NLP_Sol[5]
            C_3_sol = NLP_Sol[6]
            
            T_ave_sol = NLP_Sol[7:(N+1)+7]
            T_wall_sol = NLP_Sol[(N+1)+7:2*(N+1)+7]
            y_sol = NLP_Sol[2*(N+1)+7:2*(N+1)+7+N]
    
        elif (Type_System_Model == 4): # Four State

            ## Initial Setup

            # State/Parameter/Output Dimensions
            State_n = n_State ##-----------------------------------
            Parameter_n = n_Parameter ##-----------------------------------
            Output_n = n_Output ##-----------------------------------
            Input_n = n_Input  ##-----------------------------------

            # Initial Filter stae mean/covariance - as one state
            T_ave_ini_model = y_measured
            T_wall_ini_model = y_measured
            T_attic_ini_model = y_measured
            T_im_ini_model = y_measured

            y_ini_model = y_measured

            #parameters

            R_win_ini_model = Theta_Initial[0]
            R_w2_ini_model = Theta_Initial[1]
            R_attic_ini_model = Theta_Initial[2]
            R_im_ini_model = Theta_Initial[3]
            R_roof_ini_model = Theta_Initial[4]
            C_in_ini_model = Theta_Initial[5]
            C_w_ini_model = Theta_Initial[6]
            C_attic_ini_model = Theta_Initial[7]
            C_im_ini_model = Theta_Initial[8]
            C_1_ini_model = Theta_Initial[9]
            C_2_ini_model = Theta_Initial[10]
            C_3_ini_model = Theta_Initial[11]

            # Creating Infinity
            Infinity = np.inf


            ## Creating Optimization Variables

            # State Variables
            T_ave = SX.sym('T_ave',N+1,1)
            T_wall = SX.sym('T_ave',N+1,1)
            T_attic = SX.sym('T_attic',N+1,1)
            T_im = SX.sym('T_im',N+1,1)

            #Output Variable
            y_l = SX.sym('y_l',N,1)

            ## Getting total time steps
            N = y_measured.shape[0]

            # Parameter Variables
            R_win = SX.sym('R_win',1,1)
            R_w2 = SX.sym('R_w2',1,1)
            R_attic = SX.sym('R_attic',1,1)
            R_im = SX.sym('R_im',1,1)
            R_roof = SX.sym('R_roof',1,1)

            C_in = SX.sym('C_in',1,1)
            C_w = SX.sym('C_w',1,1)
            C_attic = SX.sym('C_attic',1,1)
            C_im = SX.sym('C_im',1,1)

            C_1 = SX.sym('C_1',1,1)
            C_2 = SX.sym('C_2',1,1)
            C_3 = SX.sym('C_3',1,1)

            # System Matrix
            A_matrix = SX.sym('A_matrix',State_n,State_n)

            A_matrix[0,0] = (-(1/C_in)*(1/(R_w2/2) + 1/R_attic + 1/R_im + 1/R_win))
            A_matrix[0,1] = 1/(C_in*(R_w2/2))
            A_matrix[0,2] = 1/(C_in*R_attic)
            A_matrix[0,3] = 1/(C_in*R_im)

            A_matrix[1,0] = 1/(C_w*(R_w2/2))
            A_matrix[1,1] = -2/(C_w*(R_w2/2))
            A_matrix[1,2] = 0
            A_matrix[1,3] = 0

            A_matrix[2,0] = 1/(C_attic*R_attic)
            A_matrix[2,1] = 0
            A_matrix[2,2] = -((1/(C_attic*R_roof)) + (1/(C_attic*R_attic)))
            A_matrix[2,3] = 0

            A_matrix[3,0] = 1/(C_im*R_im)
            A_matrix[3,1] = 0
            A_matrix[3,2] = 0
            A_matrix[3,3] = -1/(C_im*R_im)

            # System Constants
            C_matrix = DM(1,State_n)
            C_matrix[:,:] = np.hstack((np.array([[1]]),np.zeros((1,State_n-1)))) #np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

            #Creating input matrix, i.e B - [T_sol_r[ii], T_sol_w[ii], T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infil[ii]]
            B_matrix = SX.sym('B_matrix',State_n,Input_n)

            B_matrix[0,0] = 0
            B_matrix[0,1] = 0
            B_matrix[0,2] = 1/(C_in*R_win)
            B_matrix[0,3] = C_1/C_in
            B_matrix[0,4] = C_2/C_in
            B_matrix[0,5] = 0
            B_matrix[0,6] = 1/C_in
            B_matrix[0,7] = 1/C_in


            B_matrix[1,0] = 0
            B_matrix[1,1] = 1/(C_w*(R_w2/2))
            B_matrix[1,2] = 0
            B_matrix[1,3] = 0
            B_matrix[1,4] = 0
            B_matrix[1,5] = 0
            B_matrix[1,6] = 0
            B_matrix[1,7] = 0

            B_matrix[2,0] = 1/(C_attic*R_roof)
            B_matrix[2,1] = 0
            B_matrix[2,2] = 0
            B_matrix[2,3] = 0
            B_matrix[2,4] = 0
            B_matrix[2,5] = 0
            B_matrix[2,6] = 0
            B_matrix[2,7] = 0


            B_matrix[3,0] = 0
            B_matrix[3,1] = 0
            B_matrix[3,2] = 0
            B_matrix[3,3] = 0
            B_matrix[3,4] = 0
            B_matrix[3,5] = C_3/C_im
            B_matrix[3,6] = 0
            B_matrix[3,7] = 0

            ## Constructing the Cost Function

            # Cost Function Development
            CostFunction = 0

            ## Constructing the Constraints

            # Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
            T_ave_lb = []
            T_ave_ub = []

            T_wall_lb = []
            T_wall_ub = []

            T_attic_lb = []
            T_attic_ub = []

            T_im_lb = []
            T_im_ub = []

            y_lb = []
            y_ub = []

            R_win_lb = [0]
            R_win_ub = [Infinity]

            R_w2_lb = [0]
            R_w2_ub = [Infinity]

            R_attic_lb = [0]
            R_attic_ub = [Infinity]

            R_im_lb = [0]
            R_im_ub = [Infinity]

            R_roof_lb = [0]
            R_roof_ub = [Infinity]

            C_in_lb = [0]
            C_in_ub = [Infinity]

            C_w_lb = [0]
            C_w_ub = [Infinity]

            C_attic_lb = [0]
            C_attic_ub = [Infinity]

            C_im_lb = [0]
            C_im_ub = [Infinity]

            C_1_lb = [0]
            C_1_ub = [Infinity]

            C_2_lb = [0]
            C_2_ub = [Infinity]

            C_3_lb = [0]
            C_3_ub = [Infinity]       


            Eq_x_lb = []
            Eq_y_lb = []

            Eq_x_ub = []
            Eq_y_ub = []

            Eq_x = []
            Eq_y = []

            # FOR LOOP: For each time step
            for ii in range(N):

                # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
                CostFunction += (y_l[ii]-y_measured[ii])**2

                ## State/Covariance Equations - Formulation

                # Creating State Vector
                x_k_1 = SX.sym('x_k_1',State_n,1)
                x_k = SX.sym('x_k',State_n,1)

                x_k_1[0,0] = T_ave[ii+1]
                x_k_1[1,0] = T_wall[ii+1]
                x_k_1[2,0] = T_attic[ii+1]
                x_k_1[3,0] = T_im[ii+1]

                x_k[0,0] = T_ave[ii]
                x_k[1,0] = T_wall[ii]
                x_k[2,0] = T_attic[ii]
                x_k[3,0] = T_im[ii]

                #Creating input vector - U

                U_vector = DM(Input_n,1)
                U_vector[:,:] = np.reshape(np.array([T_sol_r[ii], T_sol_w[ii], T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infil[ii]]), (Input_n,1))

                # State Equation
                x_Eq = -x_k_1 + x_k + ts*(A_matrix @ x_k + B_matrix @ U_vector)
                y_Eq = -y_l[ii] + C_matrix @ x_k #D matrix only for feed forward system


                # Adding current equations to Equation List
                Eq_x += [x_Eq[0,0], x_Eq[0,1], x_Eq[1,0], x_Eq[1,1]]
                Eq_y += [y_Eq]


                # Adding Equation Bounds, [0,0] for 2 equations
                Eq_x_lb += [0,0,0,0]
                Eq_x_ub += [0,0,0,0]

                Eq_y_lb += [0]
                Eq_y_ub += [0]


                # Adding Variable Bounds
                T_ave_lb += [-Infinity]
                T_ave_ub += [Infinity]

                T_wall_lb += [-Infinity]
                T_wall_ub += [Infinity]

                T_attic_lb += [-Infinity]
                T_attic_ub += [Infinity]

                T_im_lb += [-Infinity]
                T_im_ub += [Infinity]

                y_lb += [-Infinity]
                y_ub += [Infinity]

            ## Adding Variable Bounds - For (N+1)th Variable
            T_ave_lb += [-Infinity]
            T_ave_ub += [Infinity]

            T_wall_lb += [-Infinity]
            T_wall_ub += [Infinity]

            T_attic_lb += [-Infinity]
            T_attic_ub += [Infinity]

            T_im_lb += [-Infinity]
            T_im_ub += [Infinity]

            ## Constructing NLP Problem

            # Creating Optimization Variable: x
            x = vcat([R_win, R_w2, R_attic, R_im, R_roof, C_in, C_w, C_attic, C_im, C_1, C_2, C_3, T_ave, T_wall, T_attic,T_im,y_l])

            # Creating Cost Function: J
            J = CostFunction

            # Creating Constraints: g
            g = vertcat(*Eq_x, *Eq_y)

            # Creating NLP Problem
            NLP_Problem = {'f': J, 'x': x, 'g': g}

            ## Constructiong NLP Solver
            NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

            ## Solving the NLP Problem

            # Creating Initial Variables
            T_ave_l_ini = (np.vstack((T_ave_ini_model, T_ave_ini_model[-1]))).tolist()
            T_wall_l_ini = (np.vstack((T_wall_ini_model, T_wall_ini_model[-1]))).tolist()
            T_attic_l_ini = (np.vstack((T_attic_ini_model, T_attic_ini_model[-1]))).tolist()
            T_im_l_ini = (np.vstack((T_im_ini_model, T_im_ini_model[-1]))).tolist()

            y_ini = y_measured.tolist()

            R_win_ini = (R_win_ini_model*np.ones((1,))).tolist()
            R_w2_ini = (R_w2_ini_model*np.ones((1,))).tolist()
            R_attic_ini = (R_attic_ini_model*np.ones((1,))).tolist()
            R_im_ini = (R_im_ini_model*np.ones((1,))).tolist()
            R_roof_ini = (R_roof_ini_model*np.ones((1,))).tolist()

            C_in_ini = (C_in_ini_model*np.ones((1,))).tolist()
            C_w_ini = (C_w_ini_model*np.ones((1,))).tolist()
            C_attic_ini = (C_attic_ini_model*np.ones((1,))).tolist()
            C_im_ini = (C_im_ini_model*np.ones((1,))).tolist()

            C1_ini = (C1_ini_model*np.ones((1,))).tolist()
            C2_ini = (C2_ini_model*np.ones((1,))).tolist()
            C3_ini = (C3_ini_model*np.ones((1,))).tolist()

            x_initial = vertcat(*R_win_ini, *R_w2_ini, *R_attic_ini, *R_im_ini, *R_roof_ini, *C_in_ini, *C_w_ini, *C_attic_ini, *C_im_ini, *C_1_ini, *C_2_ini, *C_3_ini, *T_ave_l_ini, *T_wall_l_ini, *T_attic_l_ini, *T_im_l_ini, *y_ini)

            # Creating Lower/Upper bounds on Variables and Equations
            x_lb = vertcat(*R_win_lb, *R_w2_lb, *R_attic_lb, *R_im_lb, *R_roof_lb, *C_in_lb, *C_w_lb, *C_attic_lb, *C_im_lb, *C_1_lb, *C_2_lb, *C_3_lb, *T_ave_lb, *T_wall_lb, *T_attic_lb, *T_im_lb, *y_lb)

            x_ub = vertcat(*R_win_ub, *R_w2_ub, *R_attic_ub, *R_im_ub, *R_roof_ub, *C_in_ub, *C_w_ub, *C_attic_ub, *C_im_ub, *C_1_ub, *C_2_ub, *C_3_ub, *T_ave_ub, *T_wall_ub, *T_attic_ub, *T_im_ub, *y_ub)

            G_lb = vertcat(*Eq_x_lb, *Eq_y_lb)

            G_ub = vertcat(*Eq_x_ub, *Eq_y_ub)

            # Solving NLP Problem
            NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

            #----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
            #--------------------------------------------------------------------------------------------------------------------------------------------------------#

            ## Getting the Solutions
            NLP_Sol = NLP_Solution['x'].full().flatten()

            R_win_sol = NLP_Sol[0]
            R_w2_sol = NLP_Sol[1]
            R_attic_sol = NLP_Sol[2]
            R_im_sol = NLP_Sol[3]
            R_roof_sol = NLP_Sol[4]
            C_in_sol = NLP_Sol[5]
            C_w_sol = NLP_Sol[6]
            C_attic_sol = NLP_Sol[7]
            C_im_sol = NLP_Sol[8]
            C_1_sol = NLP_Sol[9]
            C_2_sol = NLP_Sol[10]
            C_3_sol = NLP_Sol[11]

            T_ave_sol = NLP_Sol[12 : (N+1)+12]
            T_wall_sol = NLP_Sol[((N+1)+12) : 2*(N+1)+12]
            T_attic_sol = NLP_Sol[2*(N+1)+12 : 3*(N+1)+12]
            T_im_sol = NLP_Sol[3*(N+1)+12 : 4*(N+1)+12]

            y_sol = NLP_Sol[4*(N+1)+12 : 4*(N+1)+12+N]

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

            Parameter_Vector = np.reshape(np.array([R_za_sol , C_z_sol , Asol_sol ]),(n_Parameter,1))

        elif (Type_System_Model == 2):  # 2-State RC-Network

            Parameter_Vector = np.reshape(np.array([R_zw_sol, R_wa_sol, C_z_sol, C_w_sol, A1_sol, A2_sol]),(n_Parameter,1))


    elif (Type_Data_Source == 2):  # 4th Order House Thermal Model

        # IF Else Loop: Based on Type_System_Model
        if (Type_System_Model == 1):  # 1-State RC-Network

            Parameter_Vector = np.reshape(np.array([R_win_sol, C_in_sol, C_1_sol, C_2_sol, C_3_sol]),(n_Parameter,1))

        elif (Type_System_Model == 2):  # 2-State RC-Network

            Parameter_Vector = np.reshape(np.array([R_win_sol, R_w2_sol, C_in_sol, C_w_sol, C_1_sol, C_2_sol, C_3_sol]),(n_Parameter,1))

        elif (Type_System_Model == 4):  # 4-State RC-Network

            Parameter_Vector = np.reshape(np.array([R_win_sol, R_w2_sol, R_attic_sol, R_im_sol, R_roof_sol, C_in_sol, C_w_sol, C_attic_sol, C_im_sol, C_1_sol, C_2_sol, C_3_sol]),(n_Parameter,1))

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
