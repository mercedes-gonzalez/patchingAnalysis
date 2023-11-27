'''
    This set of functions:
    -reads a folder with abf files in it
    -creates a compiled info csv to use for plotting summary stats

    Mercedes Gonzalez. August 2023. 
    mercedesmg.com
    Precision Biosystems Lab | Georgia Institute of Technology
    Version Control: https://github.com/mercedes-gonzalez/patchingAnalysis
'''

import numpy as np
from numpy.core.fromnumeric import argmax
import pyabf
from scipy import *
import patchAnalysis as pa
import matplotlib.pyplot as plt
from math import cos, sin, log10
import seaborn as sns
import pandas as pd
from os import listdir, mkdir
from os.path import isfile, join, isdir
import csv


# for statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro, ttest_ind
import statistics 

## Definitions of classes and functions
class data:
    def __init__(self,time,response,command,sampleRate):
        self.time = time
        self.response = response
        self.command = command
        self.sampleRate = sampleRate
        self.numSweeps = 1
        
def abf2class(abf):
    abf.setSweep(0)
    if abf.sweepY[0] < -200:
        SF = 0.0005/.01 # Change scaling because units were messed up
        for sweepNumber in abf.sweepList:
            abf.setSweep(sweepNumber)
            if sweepNumber == 0:
                myData = data(time=abf.sweepX,response=abf.sweepY*SF,command=abf.sweepC,sampleRate=int(1/(abf.sweepX[1]-abf.sweepX[0])))
            else:
                myData.response = np.vstack((myData.response,abf.sweepY*SF))
                myData.command = np.vstack((myData.command,abf.sweepC))
                myData.numSweeps = myData.numSweeps + 1
    else:
        for sweepNumber in abf.sweepList:
            abf.setSweep(sweepNumber)
            if sweepNumber == 0:
                myData = data(time=abf.sweepX,response=abf.sweepY,command=abf.sweepC,sampleRate=int(1/(abf.sweepX[1]-abf.sweepX[0])))
            else:
                myData.response = np.vstack((myData.response,abf.sweepY))
                myData.command = np.vstack((myData.command,abf.sweepC))
                myData.numSweeps = myData.numSweeps + 1

    return myData

def lvm2class(commandFile,responseFile,memtest=0):
    # read from the lvms
    commandData = np.genfromtxt(commandFile,delimiter='\t',skip_header=23)
    responseData = np.genfromtxt(responseFile,delimiter='\t',skip_header=23)

    if memtest == 1:
        myData = data(time=responseData[:,0],response=responseData[:,1],command=commandData[:,1],sampleRate=round(1/(responseData[1,0]-responseData[0,0])))

    elif memtest == 0:
        [r,nSweeps] = responseData.shape
        for sweepNumber in range(nSweeps):
            if sweepNumber == 0:
                myData = data(time=responseData[:,0],response=responseData[:,1],command=commandData[:,1],sampleRate=round(1/(responseData[1,0]-responseData[0,0])))
            else:
                myData.response = np.vstack((myData.response,responseData[:,sweepNumber]))
                myData.command = np.vstack((myData.command,commandData[:,sweepNumber]))
                myData.numSweeps = myData.numSweeps + 1
    return myData

# Run this to read the main csv and generate csvs for each cell
def analyzeAllProtocols(main_filename,abf_path,save_path,brainslice=True): 
    # make directories for csvs and pngs
    fit_png_path = join(save_path,"fit_pngs")
    if not isdir(fit_png_path):
        mkdir(fit_png_path)

    pas_path = join(save_path,"pas_params")
    if not isdir(pas_path):
        mkdir(pas_path)

    firing_path = join(save_path,"firing_params")
    if not isdir(firing_path):
        mkdir(firing_path)

    spike_path = join(save_path,"spike_params")
    if not isdir(spike_path):
        mkdir(spike_path)

    cc_path = join(save_path,"currentclamp_pngs")
    if not isdir(cc_path):
        mkdir(cc_path)

    mt_path = join(save_path,"memtest_params")
    if not isdir(mt_path):
        mkdir(mt_path)
    
    # read main.xlsx for all cell info
    df = pd.read_excel(main_filename,dtype=str)

    for c,cell in enumerate(df.itertuples(name='Cell')): # loops through dataframe, each row is a cell. index using cell.region
        # # _____________________ MEMTEST _____________________
        # generate full file name and path for the abf
        base_fn = cell.date_num + cell.MT1.zfill(3)
        print(base_fn)
        if cell.MT1 != "na":
            full_abf_name = join(abf_path, cell.date, base_fn + '.abf')
            # print(full_abf_name)

            # read into a myData class
            abf = pa.open_myabf(full_abf_name)
            myData = abf2class(abf)

            [access_resistance, holding_current, membrane_resistance, membrane_capacitance, cell_tau, fit_error] = pa.analyzeMemtest(myData,base_fn,mt_path,verbose=0)

    
        # _____________________ passive parameters - FPC _____________________
        # generate full file name and path for the abf
        base_fn = cell.date_num + cell.FPC.zfill(3)
        if cell.FPC != "na":
            full_abf_name = join(abf_path, cell.date, base_fn + '.abf')

            # read into a myData class
            abf = pa.open_myabf(full_abf_name)
            myData = abf2class(abf)
            try:
                x = myData.response[0,:]
            except:
                print(base_fn, ' was denied')

            # Passive parameters 
            img_filename = join(save_path,"fit_pngs",base_fn + "-FPC")
            all_pas_params = pa.calc_pas_params(myData,img_filename,base_fn) # calculates passive properties of the cell
            # membrane_capacitance = all_pas_params[2,2]

            save_pas_params_df = pd.DataFrame(all_pas_params,columns = ['membrane_tau', 'input_resistance', 'membrane_capacitance', 'RMP', 'fit_err'])
            save_pas_params_df.insert(1,"MT-Cm",membrane_capacitance)
            save_pas_params_df.insert(1,"MT-Ra",access_resistance)
            save_pas_params_df.insert(1,"MT-holding",holding_current)
            save_pas_params_df.insert(1,"MT-Rm",membrane_resistance)
            save_pas_params_df.insert(1,"MT-celltau",cell_tau)
            
            if brainslice:
                save_pas_params_df.insert(0,"filename",base_fn)
                save_pas_params_df.insert(0,"strain",cell.strain)
                save_pas_params_df.insert(0,"sex",cell.sex)
                save_pas_params_df.insert(0,"hemisphere",cell.hemisphere)
                save_pas_params_df.insert(0,"region",cell.region)
                save_pas_params_df.insert(0,"cell_type",cell.cell_type)

            save_pas_params_df.to_csv(join(pas_path,base_fn+'-pas_params-FPC.csv'),index=False)
        else: 
            print(base_fn, "has NA")

        # _____________________ firing frequency - FPC _____________________
                # generate full file name and path for the abf
        base_fn = cell.date_num + cell.FPC.zfill(3)

        if cell.FPC != "na":
            full_abf_name = join(abf_path, cell.date, base_fn + '.abf')
            # print(full_abf_name)

            # read into a myData class
            abf = pa.open_myabf(full_abf_name)
            myData = abf2class(abf)

            # firing parameters
            all_firing_params = pa.calc_freq(myData,base_fn)
            save_firing_params_df = pd.DataFrame(all_firing_params,columns = ['sweep', 'current_inj', 'mean_firing_frequency'])
            save_firing_params_df.insert(0,"filename",base_fn)
            save_firing_params_df.insert(1,"membrane_capacitance",membrane_capacitance)
            save_firing_params_df.insert(1,"est_pA/pF",2*save_firing_params_df['sweep']-4) #estimated during experiment
            save_firing_params_df.insert(1,"calculated_pA/pF",save_firing_params_df['current_inj']/save_firing_params_df['membrane_capacitance']) #actually calculated
            save_firing_params_df.insert(1,"plot_pA/pF",round(save_firing_params_df['calculated_pA/pF'] / 2) * 2) # actually calculated
            save_firing_params_df.insert(1,"rounded_pA/pF",round(save_firing_params_df['calculated_pA/pF'])) # rounded only, based on calculated
            
            if brainslice: 
                save_firing_params_df.insert(1,"strain",cell.strain)
                save_firing_params_df.insert(1,"sex",cell.sex)
                save_firing_params_df.insert(1,"hemisphere",cell.hemisphere)
                save_firing_params_df.insert(1,"region",cell.region)
                save_firing_params_df.insert(1,"cell_type",cell.cell_type)

            save_firing_params_df.to_csv(join(firing_path,base_fn+'-firing_params-FPC.csv'),index=False)

            sshcr = [cell.strain, cell.sex, cell.hemisphere, cell.cell_type,cell.region]

            # individual spike params 
            all_spike_params = pa.calc_all_spike_params(myData,base_fn,spike_path,sshcr,extension='-FPC')

        # _____________________ passive params - FPU _____________________
                # generate full file name and path for the abf
        base_fn = cell.date_num + cell.FPU.zfill(3)

        if cell.FPU != "na":
            full_abf_name = join(abf_path, cell.date, base_fn + '.abf')
            # print(full_abf_name)

            # read into a myData class
            abf = pa.open_myabf(full_abf_name)
            myData = abf2class(abf)

            # Passive parameters 
            img_filename = join(save_path,"fit_pngs",base_fn + "-FPU")
            all_pas_params = pa.calc_pas_params(myData,img_filename,base_fn) # calculates passive properties of the cell
            # membrane_capacitance = all_pas_params[2,2]

            save_pas_params_df = pd.DataFrame(all_pas_params,columns = ['membrane_tau', 'input_resistance', 'membrane_capacitance', 'RMP', 'fit_err'])
            save_pas_params_df.insert(1,"MT-Cm",membrane_capacitance)
            save_pas_params_df.insert(1,"MT-Ra",membrane_resistance)
            save_pas_params_df.insert(1,"MT-holding",holding_current)
            save_pas_params_df.insert(1,"MT-Rm",membrane_resistance)
            save_pas_params_df.insert(1,"MT-celltau",cell_tau)

            if brainslice:
                save_pas_params_df.insert(0,"filename",base_fn)
                save_pas_params_df.insert(0,"strain",cell.strain)
                save_pas_params_df.insert(0,"sex",cell.sex)
                save_pas_params_df.insert(0,"hemisphere",cell.hemisphere)
                save_pas_params_df.insert(0,"region",cell.region)
                save_pas_params_df.insert(0,"cell_type",cell.cell_type)

            save_pas_params_df.to_csv(join(pas_path,base_fn+'-pas_params-FPU.csv'),index=False)
        else: 
            print(base_fn, "has NA")

        # _____________________ firing frequency - FPU _____________________
                # generate full file name and path for the abf
        base_fn = cell.date_num + cell.FPU.zfill(3)

        if cell.FPU != "na":
            full_abf_name = join(abf_path, cell.date, base_fn + '.abf')
            # print(full_abf_name)

            # read into a myData class
            abf = pa.open_myabf(full_abf_name)
            myData = abf2class(abf)

            # firing parameters
            all_firing_params = pa.calc_freq(myData,base_fn)
            save_firing_params_df = pd.DataFrame(all_firing_params,columns = ['sweep', 'current_inj', 'mean_firing_frequency'])
            save_firing_params_df.insert(0,"filename",base_fn)
            save_firing_params_df.insert(1,"membrane_capacitance",membrane_capacitance)
            save_firing_params_df.insert(1,"est_pA/pF",2*save_firing_params_df['sweep']-4) #estimated
            save_firing_params_df.insert(1,"calculated_pA/pF",save_firing_params_df['current_inj']/save_firing_params_df['membrane_capacitance']) #actually calculated
            save_firing_params_df.insert(1,"plot_pA/pF",round(save_firing_params_df['calculated_pA/pF'] / 2) * 2) #actually calculated
            save_firing_params_df.insert(1,"rounded_pA/pF",round(save_firing_params_df['calculated_pA/pF'])) # rounded only, based on calculated

            if brainslice: 
                save_firing_params_df.insert(1,"strain",cell.strain)
                save_firing_params_df.insert(1,"sex",cell.sex)
                save_firing_params_df.insert(1,"hemisphere",cell.hemisphere)
                save_firing_params_df.insert(0,"region",cell.region)
                save_firing_params_df.insert(0,"cell_type",cell.cell_type)
            save_firing_params_df.to_csv(join(firing_path,base_fn+'-firing_params-FPU.csv'),index=False)

            sshcr = [cell.strain, cell.sex, cell.hemisphere, cell.cell_type,cell.region]

            # individual spike params 
            all_spike_params = pa.calc_all_spike_params(myData,base_fn,spike_path,sshcr,extension='-FPU')

        membrane_resistance = 0.001
        membrane_capacitance = 0.001
        holding_current = 0.001
        access_resistance = 0.001

        # ________________________________________________________________________________________
       

    if 1:
        # Read the above generated csvs and compile into 1 csv
        pas_csv_list = [join(pas_path,f) for f in listdir(pas_path) if isfile(join(pas_path, f)) and f.endswith("-FPC.csv")]
        firing_csv_list = [join(firing_path,f) for f in listdir(firing_path) if isfile(join(firing_path, f)) and f.endswith("-FPC.csv")]
        spike_csv_list = [join(spike_path,f) for f in listdir(spike_path) if isfile(join(spike_path, f)) and f.endswith("-FPC.csv")]

        print('pas: ',pas_csv_list)
        print('firing: ',firing_csv_list)
        print('spike: ',spike_csv_list)
        df = pd.concat(map(pd.read_csv,spike_csv_list),ignore_index=True)
        df.to_csv(join(save_path,'compiled_spike_params-FPC.csv'),index=False)

        df = pd.concat(map(pd.read_csv,firing_csv_list),ignore_index=True)
        df.insert(5,"pApF",round(df["calculated_pA/pF"]/2)*2)
        df.to_csv(join(save_path,'compiled_firing_freq-FPC.csv'),index=False)

        df = pd.concat(map(pd.read_csv,pas_csv_list),ignore_index=True)
        df.to_csv(join(save_path,'compiled_pas_params-FPC.csv'),index=False)

        # Read the above generated csvs and compile into 1 csv
        pas_csv_list = [join(pas_path,f) for f in listdir(pas_path) if isfile(join(pas_path, f)) and f.endswith("-FPU.csv")]
        firing_csv_list = [join(firing_path,f) for f in listdir(firing_path) if isfile(join(firing_path, f)) and f.endswith("-FPU.csv")]
        spike_csv_list = [join(spike_path,f) for f in listdir(spike_path) if isfile(join(spike_path, f)) and f.endswith("-FPU.csv")]

        df = pd.concat(map(pd.read_csv,spike_csv_list),ignore_index=True)
        df.to_csv(join(save_path,'compiled_spike_params-FPU.csv'),index=False)

        df = pd.concat(map(pd.read_csv,firing_csv_list),ignore_index=True)
        df.insert(5,"pApF",round(df["calculated_pA/pF"]/2)*2)
        df.to_csv(join(save_path,'compiled_firing_freq-FPU.csv'),index=False)

        df = pd.concat(map(pd.read_csv,pas_csv_list),ignore_index=True)
        df.to_csv(join(save_path,'compiled_pas_params-FPU.csv'),index=False)
