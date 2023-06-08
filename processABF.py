'''
    This set of functions:
    -reads a folder with abf files in it
    -creates a compiled info csv to use for plotting summary stats

    Mercedes Gonzalez. June 2023. 
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
from math import cos, sin
import seaborn as sns
import pandas as pd
from os import listdir, mkdir
from os.path import isfile, join, isdir
import csv

## Definitions of classes and functions
class data:
    def __init__(self,time,response,command,sampleRate):
        self.time = time
        self.response = response
        self.command = command
        self.sampleRate = sampleRate
        self.numSweeps = 1
        
def abf2class(abf):
    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        if sweepNumber == 0:
            myData = data(time=abf.sweepX,response=abf.sweepY,command=abf.sweepC,sampleRate=int(1/(abf.sweepX[1]-abf.sweepX[0])))
        else:
            myData.response = np.vstack((myData.response,abf.sweepY))
            myData.command = np.vstack((myData.command,abf.sweepC))
            myData.numSweeps = myData.numSweeps + 1
        
    return myData

## Define location of folder with data files. 

def readABFs(abf_filepath): # Run this to read the abfs and get the passive params. 
    # define path for the pngs to check the exponential fit
    fit_png_path = join(abf_filepath,"fit_pngs")
    if not isdir(fit_png_path):
        mkdir(fit_png_path)

    # define path for csvs for each abf
    pas_path = join(abf_filepath,"pas_params")
    if not isdir(pas_path):
        mkdir(pas_path)

    firing_path = join(abf_filepath,"firing_params")
    if not isdir(firing_path):
        mkdir(firing_path)

    spike_path = join(abf_filepath,"spike_params")
    if not isdir(spike_path):
        mkdir(spike_path)

    # generate list of abfs from filepath
    abf_list = [f for f in listdir(abf_filepath) if isfile(join(abf_filepath, f)) and f.endswith(".abf")]
    print(abf_list)
    
    # for each abf, get the passive, firing, and spike params
    for c,f in enumerate(abf_list):
        base_fn = f[:-4]
        print(f)
        abf = pa.open_myabf(join(abf_filepath,f))
        myData = abf2class(abf)
        print('sweeps: ',myData.numSweeps)
        # Passive parameters 
        img_filename = join(abf_filepath,"fit_pngs",base_fn)
        all_pas_params = pa.calc_pas_params(myData,img_filename,base_fn) # calculates passive properties of the cell
        df = pd.DataFrame(all_pas_params,columns = ['filename','membrane_tau', 'input_resistance', 'membrane_capacitance', 'RMP', 'fit_err'])
        df.to_csv(join(pas_path,base_fn+'-pas_params.csv'),index=False)
        
        # firing parameters
        all_firing_params = pa.calc_freq(myData,base_fn)
        df = pd.DataFrame(all_firing_params,columns = ['filename','sweep', 'current_inj', 'mean_firing_frequency'])
        df.insert(1,"pA/pF",2*df['sweep']-2)
        df.to_csv(join(firing_path,base_fn+'-firing_params.csv'),index=False)

        # individual spike params 
        all_spike_params = pa.calc_all_spike_params(myData,base_fn,spike_path)

    # Read the above generated csvs and compile into 1 csv
    pas_csv_list = [join(pas_path,f) for f in listdir(pas_path) if isfile(join(pas_path, f)) and f.endswith(".csv")]
    firing_csv_list = [join(firing_path,f) for f in listdir(firing_path) if isfile(join(firing_path, f)) and f.endswith(".csv")]
    spike_csv_list = [join(spike_path,f) for f in listdir(spike_path) if isfile(join(spike_path, f)) and f.endswith(".csv")]

    df = pd.concat(map(pd.read_csv,spike_csv_list),ignore_index=True)
    df.to_csv(join(abf_filepath,'compiled_spike_params.csv'),index=False)

    df = pd.concat(map(pd.read_csv,firing_csv_list),ignore_index=True)
    df.to_csv(join(abf_filepath,'compiled_firing_freq.csv'),index=False)

    df = pd.concat(map(pd.read_csv,pas_csv_list),ignore_index=True)
    df.to_csv(join(abf_filepath,'compiled_pas_params.csv'),index=False)

def makePatchStatsFigs(abf_filepath):
    # plot formatting
    sns.set_theme(style="whitegrid")
    color = ['red','blue','green','gray']
    def setProps(color):
        PROPS = {
            'boxprops':{'facecolor':'white', 'edgecolor':color},
            'medianprops':{'color':color},
            'whiskerprops':{'color':color},
            'capprops':{'color':color}
            }
        return PROPS
    if 0: # run this to plot current vs firing freq
        alldata = pd.read_csv(join(abf_filepath,'compiled_firing_freq.csv'))
        selectdata = alldata.loc[(alldata['pA/pF'] <= 30) & (alldata['pA/pF'] > 0)]

        metric = "mean_firing_frequency"
        PROPS = setProps('black')
        ax = sns.boxplot(x='pA/pF',y=metric,data=selectdata,**PROPS)
        ax.set(xlabel="Current Injection (pA/pF)",ylabel="Mean Firing Frequency (Hz)")

        plt.tight_layout()
        plt.show()

    if 1: # plot a boxplot for a passive param
        alldata = pd.read_csv(join(abf_filepath,'june5and6_pasparams.csv'))

        PROPS = setProps('black')
        fig, axs = plt.subplots(ncols=4)
        fig.set_size_inches(9, 3)
        w = .2
        sns.boxplot(y="RMP",data=alldata,**PROPS,width=w,ax=axs[0])
        axs[0].set(ylabel="resting membrane potential (mV)")
        sns.boxplot(y="membrane_tau",data=alldata,**PROPS,width=w,ax=axs[1])
        axs[1].set(ylabel="tau (ms)")
        sns.boxplot(y="input_resistance",data=alldata,**PROPS,width=w,ax=axs[2])
        axs[2].set(ylabel="membrane resistance (M$\Omega$)")
        sns.boxplot(y="membrane_capacitance",data=alldata,**PROPS,width=w,ax=axs[3])
        axs[3].set(ylabel="membrane capacitance (pF)")

        plt.tight_layout()
        plt.show()


    if 0: # run this to plot singular boxplot
        alldata = pd.read_csv(join(csv_path,'properties.csv'))

        metric = "" #"holding (pA)"
        PROPS = setProps('black')
        fig = plt.figure(figsize=(2,3))
        ax = sns.boxplot(y=metric,data=alldata*100,**PROPS)
        plt.ylim([0,400])
        plt.tight_layout()
        # ax = sns.swarmplot(x='APnum',y='AP peak',data=inj1,color='gray')
        # ax.set_ylim([0,50])
        plt.show()

    if 0: # run this to plot summary stats on ap firing
        alldata = pd.read_csv(join(abf_filepath,'compiled_spike_params.csv'))

        APnums0 = alldata[alldata['APnum']<21]
        APnums = APnums0[APnums0['APnum'] > 0]
        inj1 = APnums[APnums['pA/pF'] == 16]
        inj2 = APnums[APnums['pA/pF'] == 20]
        inj3 = APnums[APnums['pA/pF'] == 30]

        metric = "dV/dt max"
        PROPS = setProps('black')
        ax = sns.boxplot(x='APnum',y=metric,data=inj1,**PROPS)
        # ax.set_ylim([0,50])

        PROPS = setProps('skyblue')
        ax = sns.boxplot(x='APnum',y=metric,data=inj2,**PROPS)

        PROPS = setProps('slateblue')
        ax = sns.boxplot(x='APnum',y=metric,data=inj3,**PROPS)

        # plt.legend(labels=['10','12','16'])
        # ax = sns.swarmplot(x='APnum',y='AP peak',data=inj1,color='gray')

        plt.tight_layout()
        plt.show()

    if 0:
        alldata = pd.read_csv(join(abf_filepath,'compiled_spike_params.csv'))

        APnumselect = alldata[alldata['APnum'].isin([1,10,20])]
        selectcurrent = APnumselect[APnumselect['pA/pF']==30]

        fig = plt.figure(figsize=(3,3))
        PROPS = {
        'boxprops':{'facecolor':'none', 'edgecolor':'black'},
        'medianprops':{'color':'black'},
        'whiskerprops':{'color':'black'},
        'capprops':{'color':'black'},
        'width':{.4}
        }

        metric = 'dV/dt max'
        ax = sns.boxplot(x='APnum',y=metric,data=selectcurrent,**PROPS)
        # ax = sns.swarmplot(x='APnum',y=metric,data=selectcurrent,color='gray')
        plt.tight_layout()
        plt.show()
