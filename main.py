"""
    Python based data analysis for current clamp and voltage clamp recordings. 

    Mercedes Gonzalez. June 2023. 
    mercedesmg.com
    Precision Biosystems Lab | Georgia Institute of Technology
    Version Control: https://github.com/mercedes-gonzalez/patchingAnalysis

"""

# my files to include
import processABF as pABF
import fluorescentImageShow as fi

# libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import argmax
import pyabf
from scipy import *
from math import cos, sin
import seaborn as sns
import pandas as pd
from os import listdir
from os.path import isfile, join
import csv

# settings for plotting
plt.rcParams.update({'font.size':10 })

# gather a list of abf files to analyze
abf_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/'
save_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/ADfigs/'

# reads all abf files in the abf_filepath and returns csvs for each file with passive params, firing params, and spike params
pABF.readABFs(abf_path,save_path)

# reads the csvs generated above and creates figures
pABF.makePatchStatsFigs("/Users/mercedesgonzalez/Dropbox (GaTech)/Research/ADfigs")
