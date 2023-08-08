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

def readABFs(abf_path,save_path): # Run this to read the abfs and get the passive params. 
    # define path for the pngs to check the exponential fit
    fit_png_path = join(save_path,"fit_pngs")
    if not isdir(fit_png_path):
        mkdir(fit_png_path)

    # define path for csvs for each abf
    pas_path = join(save_path,"pas_params")
    if not isdir(pas_path):
        mkdir(pas_path)

    firing_path = join(save_path,"firing_params")
    if not isdir(firing_path):
        mkdir(firing_path)

    spike_path = join(save_path,"spike_params")
    if not isdir(spike_path):
        mkdir(spike_path)

    # get mouse strain/hemisphere/sex information from csv to apply as a new column for each recording:
    info_csv_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/ADfigs/info.csv'
    info_df = pd.read_csv(info_csv_path)
    
    # generate list of folders first
    dates_list = [f for f in listdir(abf_path) if isdir(join(abf_path,f))]

    for day_num,datestr in enumerate(dates_list):
        print(datestr)
        # generate list of abfs from filepath
        abf_list = [f for f in listdir(join(abf_path,datestr)) if isfile(join(abf_path,datestr,f)) and f.endswith(".abf")]
    
        # for each abf, get the passive, firing, and spike params
        for c,abfstr in enumerate(abf_list):
            # get mouse strain/sex/hemisphere info from info_df
            search_datestr =int(abfstr[:-7])
            print('\t',abfstr)
            if search_datestr in info_df['date'].values:
                mouse_info = info_df[info_df['date'] == search_datestr]
                strain = mouse_info.iloc[0]['strain']
                hemisphere = mouse_info.iloc[0]['hemisphere']
                sex = mouse_info.iloc[0]['sex']
                ssh = [strain,sex,hemisphere]

            else:
                print("Did not find ",search_datestr," in info csv.")
                break

            # get data from abf
            base_fn = abfstr[:-4]
            abf = pa.open_myabf(join(abf_path,datestr,abfstr))
            myData = abf2class(abf)
            
            # Passive parameters 
            img_filename = join(save_path,"fit_pngs",base_fn)
            all_pas_params = pa.calc_pas_params(myData,img_filename,base_fn) # calculates passive properties of the cell
            df = pd.DataFrame(all_pas_params,columns = ['filename','membrane_tau', 'input_resistance', 'membrane_capacitance', 'RMP', 'fit_err'])
            df.insert(1,"strain",strain)
            df.insert(1,"sex",sex)
            df.insert(1,"hemisphere",hemisphere)
            df.to_csv(join(pas_path,base_fn+'-pas_params.csv'),index=False)
            
            # firing parameters
            all_firing_params = pa.calc_freq(myData,base_fn)
            df = pd.DataFrame(all_firing_params,columns = ['filename','sweep', 'current_inj', 'mean_firing_frequency'])
            df.insert(1,"pA/pF",2*df['sweep']-2)
            df.insert(1,"strain",strain)
            df.insert(1,"sex",sex)
            df.insert(1,"hemisphere",hemisphere)
            df.to_csv(join(firing_path,base_fn+'-firing_params.csv'),index=False)
            print(join(firing_path,base_fn+'-firing_params.csv'))
            # individual spike params 
            all_spike_params = pa.calc_all_spike_params(myData,base_fn,spike_path,ssh)

    # Read the above generated csvs and compile into 1 csv
    pas_csv_list = [join(pas_path,f) for f in listdir(pas_path) if isfile(join(pas_path, f)) and f.endswith(".csv")]
    firing_csv_list = [join(firing_path,f) for f in listdir(firing_path) if isfile(join(firing_path, f)) and f.endswith(".csv")]
    spike_csv_list = [join(spike_path,f) for f in listdir(spike_path) if isfile(join(spike_path, f)) and f.endswith(".csv")]

    df = pd.concat(map(pd.read_csv,spike_csv_list),ignore_index=True)
    df.to_csv(join(save_path,'compiled_spike_params.csv'),index=False)

    df = pd.concat(map(pd.read_csv,firing_csv_list),ignore_index=True)
    df.to_csv(join(save_path,'compiled_firing_freq.csv'),index=False)

    df = pd.concat(map(pd.read_csv,pas_csv_list),ignore_index=True)
    df.to_csv(join(save_path,'compiled_pas_params.csv'),index=False)

def makePatchStatsFigs(csv_path):
    # plot formatting
    sns.set_theme(style="whitegrid")
    color = ['red','blue','green','gray']
    def setProps(color):
        PROPS = {
            'boxprops':{'facecolor':'none', 'edgecolor':color},
            'medianprops':{'color':color},
            'whiskerprops':{'color':color},
            'capprops':{'color':color}
            }
        return PROPS
    if 1: # run this to plot current vs firing freq

        alldata = pd.read_csv(join(csv_path,'compiled_firing_freq.csv'))
        selectdata = alldata.loc[(alldata['pA/pF'] <= 30) & (alldata['pA/pF'] > 0)]
        # selectdata = selectdata0.loc[(selectdata0['hemisphere'] == 'L')]

        # metric = "mean_firing_frequency"
        # PROPS = setProps('black')
        # ax = sns.boxplot(x='pA/pF',y=metric,data=selectdata,**PROPS)
        # ax.set(xlabel="Current Injection (pA/pF)",ylabel="Mean Firing Frequency (Hz)")
        
        fig, axs = plt.subplots()
        fig.set_size_inches(7,4)
        
        sns.boxplot(data=selectdata, x="pA/pF", y="mean_firing_frequency", hue="strain",linewidth=1,
               palette={"B6J": "b", "hAPP": ".85"})
        sns.stripplot(data=selectdata, x="pA/pF", y="mean_firing_frequency", hue="strain",
               palette={"B6J": "b", "hAPP": ".85"})

        axs.set(ylabel="Firing Frequency (Hz)",xlabel="Current Injection (pA/pF)")
        sns.despine(left=True)

        # female_data = selectdata[selectdata["mouse"]=="B6J"]
        # male_data = selectdata[selectdata["mouse"]=="hAPP"]

        # PROPS = setProps('slateblue')
        # ax = sns.boxplot(x='pA/pF',y="mean_firing_frequency",data=female_data,**PROPS)

        # PROPS = setProps('black')
        # ax = sns.boxplot(x='pA/pF',y="mean_firing_frequency",data=male_data,**PROPS)

        plt.tight_layout()
        plt.show()

    if 0: # plot a boxplot for a passive param
        alldata = pd.read_csv(join(csv_path,'both_compiled_pas_params.csv'))
        avg_params = []
        # average the params
        list_fn = pd.unique(alldata['filename'])

        for num,fn in enumerate(list_fn):
            print('fn',fn)
            fn_df = alldata.loc[alldata['filename'] == fn]
            print(fn_df)
            mouse = fn_df.iat[0,1]
            print('mouse=',mouse)
            tau_list = np.array(pd.unique(fn_df['membrane_tau']))
            tau = np.average(tau_list)

            cap_list = np.array(pd.unique(fn_df['membrane_capacitance']))
            capacitance = np.average(cap_list)

            in_rest_list = np.array(pd.unique(fn_df['input_resistance']))
            input_resistance = np.average(in_rest_list)

            rmp_list = np.array(pd.unique(fn_df['RMP']))
            rmp = np.average(rmp_list)

            avg_params.append([fn,mouse,tau,capacitance,input_resistance,rmp])
        
        avgdata = pd.DataFrame(avg_params, columns =['filename','mouse', 'membrane_tau','membrane_capacitance','input_resistance','RMP'])


        PROPS = setProps('black')
        fig, axs = plt.subplots(ncols=4)
        fig.set_size_inches(9,3)
        w = .2
        sns.boxplot(y="RMP",x="mouse",data=avgdata,**PROPS,width=w,ax=axs[0])
        axs[0].set(ylabel="resting membrane potential (mV)",xlabel="Strain")
        sns.boxplot(y="membrane_tau",x="mouse",data=avgdata,**PROPS,width=w,ax=axs[1])
        axs[1].set(ylabel="tau (ms)",xlabel="Strain")
        sns.boxplot(y="input_resistance",x="mouse",data=avgdata,**PROPS,width=w,ax=axs[2])
        axs[2].set(ylabel="membrane resistance (M$\Omega$)",xlabel="Strain")
        sns.boxplot(y="membrane_capacitance",x="mouse",data=avgdata,**PROPS,width=w,ax=axs[3])
        axs[3].set(ylabel="membrane capacitance (pF)",xlabel="Strain")

        plt.tight_layout()
        plt.show()

    if 0: # run this to plot summary stats on ap firing
        alldata = pd.read_csv(join(csv_path,'june5and6_compiled_spike_params.csv'))

        APnums0 = alldata[alldata['APnum']<21]
        APnums = APnums0[APnums0['APnum'] > 0]
        inj1 = APnums[APnums['pA/pF'] == 16]
        inj2 = APnums[APnums['pA/pF'] == 20]
        inj3 = APnums[APnums['pA/pF'] == 30]

        metric = "AP peak"
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
        alldata = pd.read_csv(join(csv_path,'compiled_spike_params.csv'))
        APnumselect = alldata[alldata['APnum'].isin([1,10,20])]
        selectcurrent = APnumselect[APnumselect['pA/pF']==6]

        fig = plt.figure(figsize=(4,3))
        PROPS = {
        'boxprops':{'facecolor':'none', 'edgecolor':'black'},
        'medianprops':{'color':'black'},
        'whiskerprops':{'color':'black'},
        'capprops':{'color':'black'},
        'width':{.4}
        }

        metric = 'dV/dt max'
        ax = sns.boxplot(x='APnum',y=metric,hue = 'Sex',data=selectcurrent,
                         palette={"m": "b", "f": ".85"})
        ax.set(ylabel="dV/dt max",xlabel="AP Num")
        plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)

        # ax = sns.swarmplot(x='APnum',y=metric,data=selectcurrent,color='gray')
        plt.tight_layout()
        plt.show()
