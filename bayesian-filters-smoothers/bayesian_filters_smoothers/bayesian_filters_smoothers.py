# -*- coding: utf-8 -*-

# Importing external modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing internal modules

''' Bayesian Filters and Smoothers

This module implements the following Bayesian Filters and Smoothers:

    Kalman Filter/Smoother
    Extended Kalman Filter/Smoother
    Unscented Kalman Filter/Smoother
    Gaussian Filter/Smoother
    Particle Filter/Smoother

Classes:

    Pickler

Functions:

    dump(object, file)

Misc variables:

    __version__
'''

def addnum(num1,num2):
    """Adds two numbers.
    
    Args:
        num1 (int/float): The first number.
        num2 (int/float): The second number.

    Returns:
        num3 (int/float): Result of addition of num1 and num2.

    """
    num3 = num1+num2
    return num3
