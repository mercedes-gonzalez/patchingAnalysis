'''
    This set of functions generates the following figures: 
        -

    Mercedes Gonzalez. October 2023. 
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

# for statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro, ttest_ind
import statistics 
from statannotations.Annotator import Annotator

def makePatchStatsFigs(save_path):
    region_list = ['EC','mPFC','V1']
    cell_types_list = ['pyramidal','interneuron']

    for region in region_list:
        for cell in cell_types_list:
            generateRegionFigs(save_path,brain_region=region,cell_type=cell)

def generateRegionFigs(save_path,brain_region,cell_type):  
    # Function to remove outliers using the Z-score method for each group
    def removeOutliers(group,metric):
        mean_firing_freq = np.mean(group[metric])
        std_firing_freq = np.std(group[metric])
        return group[abs(group[metric] - mean_firing_freq) < zthreshold * std_firing_freq]

    # Function to remove outliers using the Z-score method for each group
    def removeOutliersPeak(group,metric):
        mean_firing_freq = np.mean(group[metric])
        std_firing_freq = np.std(group[metric])
        return group[abs(group[metric] - mean_firing_freq) < zthreshold * std_firing_freq]
    


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
    
    distvar = 'Y'
    threshold = 20000
    selected_pApF = 10
    RMPmin = -100
    RMPmax = 100
    zthreshold = 3 # Z-score threshold, you can adjust this based on your data

    csv_path = save_path
    filter_layers = 0
    if filter_layers:
        filter_string = '-Y2000'
    else: 
        filter_string = '-all'

    if 1: # run this to plot current vs firing freq
        alldata = pd.read_csv(join(csv_path,'compiled_firing_freq-FPC.csv'))
        alldata = alldata.loc[(alldata['region'] == brain_region) & (alldata['cell_type'] == cell_type)]
        alldata = alldata.loc[(alldata['RMP'] < RMPmax) & (alldata['RMP'] > RMPmin)]

        if filter_layers:
            alldata = alldata[alldata[distvar] > threshold]

        # Removing outliers from the 'pApF' column for each current injection group
        cleaned_df = alldata.groupby('est_pApF').apply(lambda x: removeOutliers(x, 'mean_firing_frequency')).reset_index(drop=True)

        # Print the DataFrame containing outliers
        outliers_df = alldata[~alldata.index.isin(cleaned_df.index)]

        xaxisstr = 'est_pApF'
        if cell_type == 'interneuron':
            max_papf = 30
        elif cell_type == 'pyramidal':
            max_papf = 20

        selectdata = cleaned_df.loc[(cleaned_df[xaxisstr] <= max_papf) & (cleaned_df[xaxisstr] >=0) & (np.mod(cleaned_df[xaxisstr],2)==0)]
        hAPPdata = selectdata.loc[(selectdata['strain'] == 'hAPPKI')]
        B6Jdata = selectdata.loc[(selectdata['strain'] == 'B6')]
                                 
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

        # # ___________________ between all groups stats _________________________
        # # Test groups for normality first.
        # stats = []
        # grouped = selectdata.groupby([xaxisstr, 'strain'])
        # for group_name, group_data in grouped['mean_firing_frequency']:
        #     shapiro_stat, shapiro_p_value = shapiro(group_data)
        #     nsamples = len(group_data)
        #     # print(f"Group: {group_name}, N: {nsamples}, Shapiro-Wilk Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p_value:.4f}")
        #     stats.append([group_name[0],group_name[1],nsamples,shapiro_stat,shapiro_p_value])
        
        # with open(join(save_path,"stats","firing_normality_"+brain_region+filter_string+cell_type".csv"), "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerow([xaxisstr,'strain','numSamples','SW-stat','p-value'])
        #     writer.writerows(stats)
        
        # _____________ ANOVA statistics just sex and strain ____________________-
        region_path = join(save_path,brain_region)
        if not isdir(region_path):
            mkdir(region_path)

        p_list = []
        unique_pApF_values = selectdata[xaxisstr].unique().tolist()
        for p,inj in enumerate(unique_pApF_values): #get unique pApF and do the stats for each injection value
            current_inj_df = selectdata.loc[(selectdata[xaxisstr] == inj)]
            model = ols('mean_firing_frequency ~ C(strain)',data=current_inj_df).fit()
            result = sm.stats.anova_lm(model,typ=2)
            result.to_csv(join(save_path,brain_region,str(inj)+brain_region+'.csv'), index=True)
            pvalue = result.loc['C(strain)', 'PR(>F)']

            if pvalue >= .05:
                p_list.append(" ")
            elif pvalue < 0.05:
                p_list.append("*")
            # elif pvalue > 0.01:
            #     p_list.append("*")
            # elif pvalue > 0.001:
            #     p_list.append("**")
            # elif pvalue > 0.0001:
            #     p_list.append("***")
            # else:
            #     p_list.append("****")


        # # ___________________ hAPP vs B6 stats _________________________
        # stats = []
        # grouped = selectdata.groupby(['strain',xaxisstr])
        # for group_name, group_data in grouped['mean_firing_frequency']:
        #     shapiro_stat, shapiro_p_value = shapiro(group_data)
        #     nsamples = len(group_data)
        #     print(f"Group: {group_name}, N: {nsamples}, Shapiro-Wilk Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p_value:.4f}")
        #     stats.append([group_name[0],group_name[1],nsamples,shapiro_stat,shapiro_p_value])
        
        # with open(join(region_path,"firing_stats-"+brain_region+filter_string+cell_type+"-normality.csv"), "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['strain','current_inj','numSamples','SW-stat','p-value'])
        #     writer.writerows(stats)
        # _______________________________________________________________

        if 0: 
            plt.figure()
            plt.scatter(pd.to_numeric(hAPPdata['X']),pd.to_numeric(hAPPdata['Y']))
            plt.show()
        if 0:
            plt.figure()

            colormin=-1
            colormax = max(pd.to_numeric(B6Jdata[distvar]))
            colormap = "rainbow"

            nums = np.random.uniform(-1,1,len(hAPPdata['est_pApF']))
            s = plt.scatter(pd.to_numeric(hAPPdata['est_pApF'])+nums/2,
                            pd.to_numeric(hAPPdata['mean_firing_frequency']),
                            c=pd.to_numeric(hAPPdata[distvar]),
                            vmin=colormin,
                            vmax=colormax,
                            cmap=colormap)
            plt.title("hAPP " +brain_region)
            plt.xlabel("Current Injection (pApF)")
            plt.ylabel("Mean firing frequency (Hz)")
            plt.ylim([0,250])
            plt.colorbar(label=distvar)

            plt.figure()
            nums = np.random.uniform(-1,1,len(B6Jdata['est_pApF']))
            s = plt.scatter(pd.to_numeric(B6Jdata['est_pApF'])+nums/2,
                            pd.to_numeric(B6Jdata['mean_firing_frequency']),
                            c=pd.to_numeric(B6Jdata[distvar]),
                            vmin=colormin,
                            vmax=colormax,
                            cmap=colormap)
            plt.title("B6J "+brain_region)
            plt.xlabel("Current Injection (pApF)")
            plt.ylabel("Mean firing frequency (Hz)")
            plt.ylim([0,250])
            plt.colorbar(label=distvar)
            plt.show()
    
        # Plotting the error bar plot
        fig, axs = plt.subplots()
        fig.set_size_inches(4,4)
        
        lw = 2
        ms = 5
        plt.errorbar(current_injection_values_hAPP, mean_values_hAPP, yerr=sem_values_hAPP, color='royalblue',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        plt.errorbar(current_injection_values_B6J, mean_values_B6J, yerr=sem_values_B6J, color='k',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        
        positions1 = np.array(mean_values_hAPP) + np.array(sem_values_hAPP)
        positions2 = np.array(mean_values_B6J) + np.array(sem_values_B6J)

        if len(positions1) > len(positions2):
            positions2 = np.append(positions2,np.zeros((len(positions1) - len(positions2))))
        elif len(positions1) < len(positions2):
            positions1 = np.append(positions1,np.zeros((len(positions2) - len(positions1))))
        positions = np.maximum(positions1,positions2)

        # plot p value indicators
        for i, label in enumerate(p_list):
            plt.annotate(label,(unique_pApF_values[i],positions[i]),xytext=(0,10),textcoords="offset points", ha='center')
        
        plt.xlabel('Current Injection (pApF)')
        plt.ylabel('Mean Firing Frequency (Hz)')

        if brain_region == 'interneuron':
            plt.ylim([0,250])
        elif brain_region == 'pyramidal':
            plt.ylim([0,100])

        plt.xlim([0,max_papf+2])
        plt.legend(['hAPP','B6J'])

        try:
            plt.title(brain_region + ': hAPP ('+ str(hAPPdata.groupby(xaxisstr)['mean_firing_frequency'].size()[0]) +'), B6 (' + str(B6Jdata.groupby(xaxisstr)['mean_firing_frequency'].size()[0]) +')')
        except:
            print('no title')
        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region+cell_type+'firing.svg'),dpi=300,format='svg')
        plt.clf()

        if 0:
            # normality violin plots - one plot with B6 vs hAPP
            fig3, axs3 = plt.subplots(1)
            fig3.set_size_inches(16,8)

            sns.violinplot(data=selectdata,x='est_pApF',y='mean_firing_frequency',hue='strain',split=True,ax=axs3).set(title="hAPP vs B6J")
            plt.tight_layout()
        if 0:
            # normality violin plots - two plots, for M and F 
            fig3, axs3 = plt.subplots(2)
            fig3.set_size_inches(10,8)
            maledata = selectdata[selectdata['sex'] == 'M']
            femaledata = selectdata[selectdata['sex'] == 'F']

            sns.violinplot(data=maledata,x='pApF',y='mean_firing_frequency',hue='strain',split=True,ax=axs3[0]).set(title="Male")
            sns.violinplot(data=femaledata,x='pApF',y='mean_firing_frequency',hue='strain',split=True,ax=axs3[1]).set(title="Female")

            plt.tight_layout()
        if 0:
            # normality violin plots - two plots, for hAPP and B6
            fig3, axs3 = plt.subplots(2)
            fig3.set_size_inches(10,8)
            hAPPdata = selectdata[selectdata['strain'] == 'hAPP']
            B6Jdata = selectdata[selectdata['strain'] == 'B6J']

            sns.violinplot(data=hAPPdata,x='pApF',y='mean_firing_frequency',hue='sex',split=True,ax=axs3[0]).set(title="hAPP")
            sns.violinplot(data=B6Jdata,x='pApF',y='mean_firing_frequency',hue='sex',split=True,ax=axs3[1]).set(title="B6J")

            plt.tight_layout()


    if 1: # plot a boxplot for a passive param
        alldata = pd.read_csv(join(csv_path,'compiled_pas_params-FPC.csv'))
        alldata = alldata.loc[(alldata['region'] == brain_region) & (alldata['cell_type'] == cell_type)]
        alldata = alldata.loc[(alldata['MT-celltau'] > 0) & (alldata['MT-Rm'] >0)]
        alldata = alldata.loc[(alldata['RMP'] < RMPmax) & (alldata['RMP'] > RMPmin)]

        # ___________________________ SUBFUNCTIONS _______________________________
        def unpairedTTest(avgdata,measured_metric):
            hAPP = avgdata[avgdata['strain']=='hAPPKI']
            B6J = avgdata[avgdata['strain']=='B6']

            group1 = hAPP[measured_metric]
            group2 = B6J[measured_metric]

            stat, pvalue = ttest_ind(group1,group2,equal_var = 0)
            nGroup1 = len(group1)
            nGroup2 = len(group2)

            return [measured_metric,nGroup1,nGroup2,stat,pvalue]

        def unpairedTTestAPPeak(avgdata,measured_metric):
            hAPP = avgdata[avgdata['strain']=='hAPPKI']
            B6J = avgdata[avgdata['strain']=='B6']

            group1 = hAPP[measured_metric]
            group2 = B6J[measured_metric]

            stat, pvalue = ttest_ind(group1,group2,equal_var = 0)

            return pvalue
        
        def makeBoxplot(metric,metric_str,axis_num,pvalues):
            sns.boxplot(y=metric,x="strain",data=avgdata,**PROPS,width=w,ax=axs[axis_num], order = plot_order)
            sns.swarmplot(y=metric,x="strain",hue=huestr,data=avgdata,zorder=.5,ax=axs[axis_num],palette=palstr,size=ms, order = plot_order)
            axs[axis_num].set(ylabel=metric_str,xlabel="")
            axs[axis_num].get_legend().remove()

            # set up annotator for p values to be plotted automatically
            # https://levelup.gitconnected.com/statistics-on-seaborn-plots-with-statannotations-2bfce0394c00
            # https://github.com/trevismd/statannotations/tree/master

            pairs = [('B6','hAPPKI')]
            hue_plot_params = {'data': avgdata, 'x': 'strain','y': metric,}
            annotator = Annotator(axs[axis_num], pairs, **hue_plot_params)
            formatted_pvalues = [f'p={pvalue:.2g}' for pvalue in pvalues]
            annotator.set_custom_annotations(formatted_pvalues)
            annotator.annotate()

            return
        # ________________________________________________________________________

        if filter_layers:
            alldata = alldata[alldata[distvar] > threshold]

        metrics = ['membrane_capacitance','membrane_tau','input_resistance','MT-holding','RMP']

        for metric in metrics:
            cleaned_df = alldata.groupby('strain').apply(lambda x: removeOutliers(x, metric)).reset_index(drop=True)

        # Print the DataFrame containing outliers
        outliers_df = alldata[~alldata.index.isin(cleaned_df.index)]
        print("Outliers passive:",len(outliers_df))

        avg_params = []
        # average the params
        list_fn = pd.unique(cleaned_df['filename'])

        for num,fn in enumerate(list_fn):
            fn_df = cleaned_df.loc[cleaned_df['filename'] == fn]

            strain = fn_df.iat[0,6] # change these if add more columns
            sex = fn_df.iat[0,5]
            hemisphere = fn_df.iat[0,4]

            tau_list = np.array(pd.unique(fn_df['membrane_tau']))
            tau = np.nanmedian(tau_list)

            cap_list = np.array(pd.unique(fn_df['membrane_capacitance']))
            capacitance = np.nanmedian(cap_list)

            in_rest_list = np.array(pd.unique(fn_df['input_resistance']))
            input_resistance = np.nanmedian(in_rest_list)

            rmp_list = np.array(pd.unique(fn_df['RMP']))
            rmp = np.nanmean(rmp_list)

            holding_list = np.array(pd.unique(fn_df['MT-holding']))
            holding = np.nanmedian(holding_list)

            avg_params.append([fn,strain,sex,hemisphere,tau,capacitance,input_resistance,rmp,holding])

        avgdata = pd.DataFrame(avg_params, columns =['filename','strain','sex','hemisphere','membrane_tau','membrane_capacitance','input_resistance','RMP','holding'])

        # t test statistics
        pas_stats = []
        
        ttest_result = unpairedTTest(avgdata,'membrane_tau')
        p_tau = ttest_result[4]
        pas_stats.append(ttest_result)

        ttest_result = unpairedTTest(avgdata,'input_resistance')
        p_Rm = ttest_result[4]
        pas_stats.append(ttest_result)

        ttest_result = unpairedTTest(avgdata,'membrane_capacitance')
        p_Cm = ttest_result[4]
        pas_stats.append(ttest_result)
        
        ttest_result = unpairedTTest(avgdata,'RMP')
        p_rmp = ttest_result[4]
        pas_stats.append(ttest_result)

        ttest_result = unpairedTTest(avgdata,'holding')
        p_hold = ttest_result[4]
        pas_stats.append(ttest_result)

        with open(join(save_path,"stats","pas_stats_"+brain_region+filter_string+cell_type+".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(['metric','n (hAPP)','n (B6J)','stat','p-value'])
            writer.writerows(pas_stats)
        
        # generate plots
        PROPS = setProps('black')
        fig, axs = plt.subplots(ncols=5)
        fig.set_size_inches(10,3)
        w = .2
        ms = 4
        huestr = "strain"
        palstr = ['k','royalblue']
        plot_order = ["B6", "hAPPKI"]

        makeBoxplot(metric="RMP",metric_str="resting membrane potential (mV)",axis_num=0,pvalues=[p_rmp])
        makeBoxplot(metric="membrane_tau",metric_str="tau (ms)",axis_num=1,pvalues=[p_tau])
        makeBoxplot(metric="input_resistance",metric_str="membrane resistance (M$\Omega$)",axis_num=2,pvalues=[p_Rm])
        makeBoxplot(metric="membrane_capacitance",metric_str="membrane capacitance (pF)",axis_num=3,pvalues=[p_Cm])
        makeBoxplot(metric="holding",metric_str="holding current (pA)",axis_num=4,pvalues=[p_hold])

        plt.suptitle(brain_region + ': hAPP ('+ str(len(avgdata[avgdata['strain']=='hAPPKI'])) +'), B6 (' + str(len(avgdata[avgdata['strain']=='B6'])) +')')
        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region + filter_string + cell_type + '-pas.svg'),dpi=300,format='svg')
        plt.clf()

    if 1: # plot a boxplot for a spike params
        alldata = pd.read_csv(join(csv_path,'compiled_spike_params-FPC.csv'))
        alldata = alldata.loc[(alldata['region'] == brain_region) & (alldata['cell_type'] == cell_type)]
        alldata = alldata.loc[(alldata['RMP'] < RMPmax) & (alldata['RMP'] > RMPmin)]
        alldata = alldata.loc[(alldata['APnum'] == 0) & (alldata['pApF'] == selected_pApF)]


        # ___________________________ SUBFUNCTIONS _______________________________
        def makeBoxplotSpikeParams(metric,metric_str,data,axis_num,pvalues):
            sns.boxplot(y=metric,x="strain",data=data,**PROPS,width=w,ax=axs[axis_num], order = plot_order)
            sns.swarmplot(y=metric,x="strain",hue=huestr,data=data,zorder=.5,ax=axs[axis_num],palette=palstr,size=ms, order = plot_order)
            axs[axis_num].set(ylabel=metric_str,xlabel="")
            axs[axis_num].get_legend().remove()

            # set up annotator for p values to be plotted automatically
            # https://levelup.gitconnected.com/statistics-on-seaborn-plots-with-statannotations-2bfce0394c00
            # https://github.com/trevismd/statannotations/tree/master

            pairs = [('B6','hAPPKI')]
            hue_plot_params = {'data': data, 'x': 'strain','y': metric,}
            annotator = Annotator(axs[axis_num], pairs, **hue_plot_params)
            formatted_pvalues = [f'p={pvalue:.2g}' for pvalue in pvalues]
            annotator.set_custom_annotations(formatted_pvalues)
            annotator.annotate()

            return
        
        # ________________________________________________________________________

        if filter_layers:
            alldata = alldata[alldata[distvar] > threshold]


        metrics = ['AHP','threshold','dVdt max','AP peak','AP hwdt']

        for metric in metrics:
            # Removing outliers from the 'pApF' column for each current injection group
            cleaned_df = alldata.groupby('strain').apply(lambda x: removeOutliers(x, metric)).reset_index(drop=True)

        # Print the DataFrame containing outliers
        outliers_df = alldata[~alldata.index.isin(cleaned_df.index)]
        print("Outliers spike:",len(outliers_df))
        
        PROPS = setProps('black')
        fig, axs = plt.subplots(ncols=5)#,nrows=2)
        fig.set_size_inches(10,3)
        w = .2

        huestr = "strain"
        palstr = ['k','royalblue']
        plot_order = ["B6", "hAPPKI"]
        
        ttest_result = unpairedTTest(cleaned_df,'AP peak')
        p_peak = ttest_result[4]
        pas_stats.append(ttest_result)
 
        ttest_result = unpairedTTest(cleaned_df,'AP hwdt')
        p_hwdt = ttest_result[4]
        pas_stats.append(ttest_result)
  
        ttest_result = unpairedTTest(cleaned_df,'threshold')
        p_threshold = ttest_result[4]
        pas_stats.append(ttest_result)

        ttest_result = unpairedTTest(cleaned_df,'dVdt max')
        p_dvdt = ttest_result[4]
        pas_stats.append(ttest_result)

        ttest_result = unpairedTTest(cleaned_df,'AHP')
        p_ahp = ttest_result[4]
        pas_stats.append(ttest_result)

        makeBoxplotSpikeParams(metric="AP peak",metric_str="AP Peak (mV)",data=cleaned_df,axis_num=0,pvalues=[p_peak])
        makeBoxplotSpikeParams(metric="AP hwdt",metric_str="AP hwdt (ms)",data=cleaned_df,axis_num=1,pvalues=[p_hwdt])
        makeBoxplotSpikeParams(metric="threshold",metric_str="threshold (mV)",data=cleaned_df,axis_num=2,pvalues=[p_threshold])
        makeBoxplotSpikeParams(metric="dVdt max",metric_str="dV/dt max (mV/s)",data=cleaned_df,axis_num=3,pvalues=[p_dvdt])
        makeBoxplotSpikeParams(metric="AHP",metric_str="AHP",data=cleaned_df,axis_num=4,pvalues=[p_ahp])

        plt.suptitle(brain_region + ': hAPP ('+ str(len(cleaned_df[cleaned_df['strain']=='hAPPKI'])) +'), B6 (' + str(len(cleaned_df[cleaned_df['strain']=='B6'])) +')')
        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region+filter_string+cell_type+'-spike.svg'),dpi=300,format='svg')
        plt.clf()

        with open(join(save_path,"stats","spike_stats_"+brain_region+filter_string+cell_type+".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(['metric','n (hAPP)','n (B6J)','stat','p-value'])
            writer.writerows(pas_stats)
    

    if 1: # run this to plot ap peak vs ap num
        alldata = pd.read_csv(join(csv_path,'compiled_spike_params-FPC.csv'))
        alldata = alldata.loc[(alldata['region'] == brain_region) & (alldata['cell_type'] == cell_type)]
        alldata = alldata.loc[(alldata['RMP'] < RMPmax) & (alldata['RMP'] > RMPmin)]


        # Removing outliers from the 'APnum' column for each current injection group
        cleaned_df = alldata.groupby(['APnum','strain']).apply(lambda x: removeOutliersPeak(x, 'AP peak')).reset_index(drop=True)

        xaxisstr = 'APnum' # independent variable, with strain too
        yaxisstr = 'AP peak' # dependent variable
        if cell_type == 'pyramidal':
            xlimit = 14
        elif cell_type == 'interneuron':
            xlimit = 30

        selectdata = cleaned_df.loc[(cleaned_df[xaxisstr] <= xlimit) & (cleaned_df[xaxisstr] >=0) & (cleaned_df['pApF'] == selected_pApF)]
        hAPPdata = selectdata.loc[(selectdata['strain'] == 'hAPPKI')]
        B6Jdata = selectdata.loc[(selectdata['strain'] == 'B6')]
                                 
        mean_firing_freq = hAPPdata.groupby(xaxisstr)[yaxisstr].mean()
        sem_firing_freq = hAPPdata.groupby(xaxisstr)[yaxisstr].sem()
        current_injection_values_hAPP = mean_firing_freq.index.tolist()
        mean_values_hAPP = mean_firing_freq.values.tolist()
        sem_values_hAPP = sem_firing_freq.values.tolist()

        mean_firing_freq = B6Jdata.groupby(xaxisstr)[yaxisstr].mean()
        sem_firing_freq = B6Jdata.groupby(xaxisstr)[yaxisstr].sem()
        current_injection_values_B6J = mean_firing_freq.index.tolist()
        mean_values_B6J = mean_firing_freq.values.tolist()
        sem_values_B6J = sem_firing_freq.values.tolist()

        # stats
        result = cleaned_df.groupby(xaxisstr).apply(lambda x: unpairedTTestAPPeak(x,yaxisstr))        

        metric = "AP peak"
        PROPS = setProps('black')

        # Plotting the error bar plot
        fig, axs = plt.subplots()
        fig.set_size_inches(4,4)

        lw = 2
        ms = 5
        
        plt.errorbar(current_injection_values_hAPP, mean_values_hAPP, yerr=sem_values_hAPP, color='royalblue',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white',label='hAPPKI')
        plt.errorbar(current_injection_values_B6J, mean_values_B6J, yerr=sem_values_B6J, color='k',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white',label='B6')

        # plot p value indicators
        pvalues = result.values[:xlimit]
        p_list = []

        for pvalue in pvalues:
            if pvalue >= .05:
                p_list.append(" ")
            elif pvalue < 0.05:
                p_list.append("*")
        
        positions1 = np.array(mean_values_hAPP) + np.array(sem_values_hAPP)
        positions2 = np.array(mean_values_B6J) + np.array(sem_values_B6J)

        if len(positions1) > len(positions2):
            positions2 = np.append(positions2,np.zeros((len(positions1) - len(positions2))))
        elif len(positions1) < len(positions2):
            positions1 = np.append(positions1,np.zeros((len(positions2) - len(positions1))))
        positions = np.maximum(positions1,positions2)

        for i, label in enumerate(p_list):
            plt.annotate(label,(current_injection_values_hAPP[i],positions[i]),xytext=(0,5),textcoords="offset points", ha='center')
        

        plt.xlabel('AP Number')
        plt.ylabel('AP Peak (mV)')

        plt.title(brain_region)

        plt.legend()

        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region+'-'+cell_type+'-APpeak.svg'),dpi=300,format='svg')
        plt.clf()