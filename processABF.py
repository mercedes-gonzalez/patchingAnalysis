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
        # print(datestr)
        # generate list of abfs from filepath
        abf_list = [f for f in listdir(join(abf_path,datestr)) if isfile(join(abf_path,datestr,f)) and f.endswith(".abf")]
    
        # for each abf, get the passive, firing, and spike params
        for c,abfstr in enumerate(abf_list):
            # get mouse strain/sex/hemisphere info from info_df
            search_datestr =int(abfstr[:-7])
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
            membrane_capacitance = all_pas_params[:,3].mean()
            df = pd.DataFrame(all_pas_params,columns = ['filename','membrane_tau', 'input_resistance', 'membrane_capacitance', 'RMP', 'fit_err'])
            df.insert(1,"strain",strain)
            df.insert(1,"sex",sex)
            df.insert(1,"hemisphere",hemisphere)
            df.to_csv(join(pas_path,base_fn+'-pas_params.csv'),index=False)
            
            # firing parameters
            all_firing_params = pa.calc_freq(myData,base_fn)
            df = pd.DataFrame(all_firing_params,columns = ['filename','sweep', 'current_inj', 'mean_firing_frequency'])
            df.insert(1,"membrane_capacitance",membrane_capacitance)
            df.insert(1,"est_pA/pF",2*df['sweep']-2) #estimated
            df.insert(1,"pA/pF",df['current_inj']/df['membrane_capacitance']) #actually calculated
            df.insert(1,"strain",strain)
            df.insert(1,"sex",sex)
            df.insert(1,"hemisphere",hemisphere)
            df.to_csv(join(firing_path,base_fn+'-firing_params.csv'),index=False)
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
        print('alldata: ',len(alldata))
        # Function to remove outliers using the Z-score method for each group
        def remove_outliers(group):
            threshold = 3  # Z-score threshold, you can adjust this based on your data
            mean_firing_freq = np.mean(group['mean_firing_frequency'])
            std_firing_freq = np.std(group['mean_firing_frequency'])
            return group[abs(group['mean_firing_frequency'] - mean_firing_freq) < threshold * std_firing_freq]

        # Removing outliers from the 'pA/pF' column for each current injection group
        cleaned_df = alldata.groupby('pA/pF').apply(remove_outliers).reset_index(drop=True)
        print('cleaned: ',len(cleaned_df))
        # Print the DataFrame containing outliers
        outliers_df = alldata[~alldata.index.isin(cleaned_df.index)]
        print("Outliers:")
        print(len(outliers_df))

        xaxisstr = 'pA/pF'
        selectdata = cleaned_df.loc[(cleaned_df['pA/pF'] <= 30) & (cleaned_df['pA/pF'] >0)]
        hAPPdata = selectdata.loc[(selectdata['strain'] == 'hAPP')]
        B6Jdata = selectdata.loc[(selectdata['strain'] == 'B6J')]
                                 
        mean_firing_freq = hAPPdata.groupby(xaxisstr)['mean_firing_frequency'].mean()
        sem_firing_freq = hAPPdata.groupby(xaxisstr)['mean_firing_frequency'].sem()
        current_injection_values_hAPP = mean_firing_freq.index.tolist()
        mean_values_hAPP = mean_firing_freq.values.tolist()
        sem_values_hAPP = sem_firing_freq.values.tolist()

        mean_firing_freq = B6Jdata.groupby(xaxisstr)['mean_firing_frequency'].mean()
        sem_firing_freq = B6Jdata.groupby(xaxisstr)['mean_firing_frequency'].sem()
        current_injection_values_B6J = mean_firing_freq.index.tolist()
        mean_values_B6J = mean_firing_freq.values.tolist()
        sem_values_B6J = sem_firing_freq.values.tolist()

        fig, axs = plt.subplots()
        fig.set_size_inches(4,4)
        
        # Plotting the error bar plot
        lw = 2
        ms = 7
        plt.errorbar(current_injection_values_hAPP, mean_values_hAPP, yerr=sem_values_hAPP, color='k',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        plt.errorbar(current_injection_values_B6J, mean_values_B6J, yerr=sem_values_B6J, color='royalblue',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        plt.xlabel('Current Injection (pA/pF)')
        plt.ylabel('Mean Firing Frequency (Hz)')
        plt.legend(['hAPP','B6J'])
        plt.tight_layout()
        # plt.show()

    if 1: # plot a boxplot for a passive param
        alldata = pd.read_csv(join(csv_path,'compiled_pas_params.csv'))
        alldata = alldata.loc[(alldata['fit_err'] <= .5) & (alldata['membrane_tau'] > 0) & (alldata['membrane_tau'] < 85)]
        
        # Function to remove outliers using the Z-score method for each group
        def remove_outliers(group,metric):
            threshold = 3  # Z-score threshold, you can adjust this based on your data
            mean_firing_freq = np.mean(group[metric])
            std_firing_freq = np.std(group[metric])
            return group[abs(group[metric] - mean_firing_freq) < threshold * std_firing_freq]

        metrics = ['membrane_tau','membrane_capacitance','input_resistance','RMP']

        for metric in metrics:
            # Removing outliers from the 'pA/pF' column for each current injection group
            cleaned_df = alldata.groupby('strain').apply(lambda x: remove_outliers(x, metric)).reset_index(drop=True)
        print("OUTLIERS: ",len(alldata)-len(cleaned_df))

        avg_params = []
        # average the params
        list_fn = pd.unique(cleaned_df['filename'])

        for num,fn in enumerate(list_fn):
            fn_df = cleaned_df.loc[cleaned_df['filename'] == fn]
            strain = fn_df.iat[0,3]
            sex = fn_df.iat[0,2]
            hemisphere = fn_df.iat[0,1]

            tau_list = np.array(pd.unique(fn_df['membrane_tau']))
            tau = np.average(tau_list)

            cap_list = np.array(pd.unique(fn_df['membrane_capacitance']))
            capacitance = np.average(cap_list)

            in_rest_list = np.array(pd.unique(fn_df['input_resistance']))
            input_resistance = np.average(in_rest_list)

            rmp_list = np.array(pd.unique(fn_df['RMP']))
            rmp = np.average(rmp_list)

            avg_params.append([fn,strain,sex,hemisphere,tau,capacitance,input_resistance,rmp])
        
        avgdata = pd.DataFrame(avg_params, columns =['filename','strain','sex','hemisphere','membrane_tau','membrane_capacitance','input_resistance','RMP'])
        print("NUM CELLS PLOTTED:",len(avgdata))

        PROPS = setProps('black')
        fig2, axs2 = plt.subplots(ncols=4)
        fig2.set_size_inches(10,3)
        w = .2
        palstr = ['orangered','royalblue']
        sns.boxplot(y="RMP",x="strain",data=avgdata,**PROPS,width=w,ax=axs2[3])
        sns.swarmplot(y="RMP",x="strain",data=avgdata,zorder=.5,ax=axs2[3],palette=palstr)
        axs2[3].set(ylabel="resting membrane potential (mV)",xlabel="")

        sns.boxplot(y="membrane_tau",x="strain",data=avgdata,**PROPS,width=w,ax=axs2[2])
        sns.swarmplot(y="membrane_tau",x="strain",data=avgdata,zorder=.5,ax=axs2[2],palette=palstr)
        axs2[2].set(ylabel="tau (ms)",xlabel="")

        sns.boxplot(y="input_resistance",x="strain",data=avgdata,**PROPS,width=w,ax=axs2[0])
        sns.swarmplot(y="input_resistance",x="strain",data=avgdata,zorder=.5,ax=axs2[0],palette=palstr)
        axs2[0].set(ylabel="membrane resistance (M$\Omega$)",xlabel="")

        sns.boxplot(y="membrane_capacitance",x="strain",data=avgdata,**PROPS,width=w,ax=axs2[1])
        sns.swarmplot(y="membrane_capacitance",x="strain",data=avgdata,zorder=.5,ax=axs2[1],palette=palstr)
        axs2[1].set(ylabel="membrane capacitance (pF)",xlabel="")

        plt.tight_layout()
        
    if 1: # plot a boxplot for a spike params
        alldata = pd.read_csv(join(csv_path,'compiled_spike_params.csv'))
        alldata = alldata.loc[(alldata['APnum'] == 0) & (alldata['sweep'] == 3)] # only first AP
        
        # Function to remove outliers using the Z-score method for each group
        def remove_outliers(group,metric):
            threshold = 3  # Z-score threshold, you can adjust this based on your data
            mean_firing_freq = np.mean(group[metric])
            std_firing_freq = np.std(group[metric])
            return group[abs(group[metric] - mean_firing_freq) < threshold * std_firing_freq]

        metrics = ['AHP','threshold','dV/dt max','AP peak']

        for metric in metrics:
            # Removing outliers from the 'pA/pF' column for each current injection group
            cleaned_df = alldata.groupby('strain').apply(lambda x: remove_outliers(x, metric)).reset_index(drop=True)
        print("OUTLIERS: ",len(alldata)-len(cleaned_df))

        PROPS = setProps('black')
        fig2, axs2 = plt.subplots(ncols=4)#,nrows=2)
        fig2.set_size_inches(10,3)
        w = .2
        palstr = ['orangered','royalblue']
        sns.boxplot(y="AP peak",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[0])
        sns.swarmplot(y="AP peak",x="strain",data=cleaned_df,zorder=.5,ax=axs2[0],palette=palstr)
        axs2[0].set(ylabel="AP peak (mV)",xlabel="")

        sns.boxplot(y="AP hwdt",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[1])
        sns.swarmplot(y="AP hwdt",x="strain",data=cleaned_df,zorder=.5,ax=axs2[1],palette=palstr)
        axs2[1].set(ylabel="AP hwdt (mV)",xlabel="")

        # sns.boxplot(y="AHP",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[2])
        # sns.swarmplot(y="AHP",x="strain",data=cleaned_df,zorder=.5,ax=axs2[2],palette=palstr)
        # axs2[2].set(ylabel="AHP",xlabel="")

        sns.boxplot(y="threshold",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[3])
        sns.swarmplot(y="threshold",x="strain",data=cleaned_df,zorder=.5,ax=axs2[3],palette=palstr)
        axs2[3].set(ylabel="threshold (pF)",xlabel="")

        sns.boxplot(y="dV/dt max",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[2])
        sns.swarmplot(y="dV/dt max",x="strain",data=cleaned_df,zorder=.5,ax=axs2[2],palette=palstr)
        axs2[2].set(ylabel="dV/dt max (mV/s)",xlabel="")

        plt.tight_layout()

    # if 1: # run this to plot ap peak vs ap num, not yet done 
    #     alldata = pd.read_csv(join(csv_path,'compiled_spike_params.csv'))

    #     # Function to remove outliers using the Z-score method for each group
    #     def remove_outliers(group):
    #         threshold = 3  # Z-score threshold, you can adjust this based on your data
    #         mean_firing_freq = np.mean(group['mean_firing_frequency'])
    #         std_firing_freq = np.std(group['mean_firing_frequency'])
    #         return group[abs(group['mean_firing_frequency'] - mean_firing_freq) < threshold * std_firing_freq]

    #     # Removing outliers from the 'pA/pF' column for each current injection group
    #     cleaned_df = alldata.groupby('pA/pF').apply(remove_outliers).reset_index(drop=True)
    #     print('cleaned: ',len(cleaned_df))
    #     # Print the DataFrame containing outliers
    #     outliers_df = alldata[~alldata.index.isin(cleaned_df.index)]
    #     print("Outliers:")
    #     print(len(outliers_df))

    #     xaxisstr = 'pA/pF'
    #     selectdata = cleaned_df.loc[(cleaned_df['pA/pF'] <= 30) & (cleaned_df['pA/pF'] >= 0)]
    #     hAPPdata = selectdata.loc[(selectdata['strain'] == 'hAPP')]
    #     B6Jdata = selectdata.loc[(selectdata['strain'] == 'B6J')]
                                 
    #     mean_firing_freq = hAPPdata.groupby(xaxisstr)['mean_firing_frequency'].mean()
    #     sem_firing_freq = hAPPdata.groupby(xaxisstr)['mean_firing_frequency'].sem()
    #     current_injection_values_hAPP = mean_firing_freq.index.tolist()
    #     mean_values_hAPP = mean_firing_freq.values.tolist()
    #     sem_values_hAPP = sem_firing_freq.values.tolist()

    #     mean_firing_freq = B6Jdata.groupby(xaxisstr)['mean_firing_frequency'].mean()
    #     sem_firing_freq = B6Jdata.groupby(xaxisstr)['mean_firing_frequency'].sem()
    #     current_injection_values_B6J = mean_firing_freq.index.tolist()
    #     mean_values_B6J = mean_firing_freq.values.tolist()
    #     sem_values_B6J = sem_firing_freq.values.tolist()


    #     APnums0 = alldata[alldata['APnum']<21]
    #     APnums = APnums0[APnums0['APnum'] > 0]
    #     inj1 = APnums[APnums['pA/pF'] == 16]
    #     inj2 = APnums[APnums['pA/pF'] == 20]
    #     inj3 = APnums[APnums['pA/pF'] == 30]

    #     metric = "AP peak"
    #     PROPS = setProps('black')
    #     ax = sns.boxplot(x='APnum',y=metric,data=inj1,**PROPS)
    #     # ax.set_ylim([0,50])

    #     PROPS = setProps('skyblue')
    #     ax = sns.boxplot(x='APnum',y=metric,data=inj2,**PROPS)

    #     PROPS = setProps('slateblue')
    #     ax = sns.boxplot(x='APnum',y=metric,data=inj3,**PROPS)

    #     plt.tight_layout()
    plt.show()
