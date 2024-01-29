"""
    Python based data analysis for current clamp and voltage clamp recordings. 

    Mercedes Gonzalez. October 2023. 
    mercedesmg.com
    Precision Biosystems Lab | Georgia Institute of Technology
    Version Control: https://github.com/mercedes-gonzalez/patchingAnalysis

"""

# my files to include
import processABF as pABF
import generateFigures as gfig
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
import time
# settings for plotting
plt.rcParams.update({'font.size':10})

start = time.time()
# define file paths for grabbing and saving data
abf_path = '/Users/mercedesgonzalez/ Dropbox (GaTech)/Research/hAPP AD Project/Data/'
save_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Figs/Spring 2024/'
main_filename = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/main/main27.xlsx'

# reads .xlsx file with WC info and makes lists instead of jus t reading all .abfs in a folder.
# then returns csvs for each file with passive params, firing params, and spike params
pABF.analyzeAllProtocols(main_filename,abf_path,save_path,brainslice=True)

# reads the csvs generated above and creates figures
# gfig.makePatchStatsFigs(save_path)

end = time.time()
print("Total time: ",end-start)
