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
    for region in region_list:
        generateRegionFigs(save_path,brain_region=region)

def generateRegionFigs(save_path,brain_region):  
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
    cell_type = 'interneuron' # interneuron or pyramidal
    threshold = 20000

    csv_path = save_path
    filter_layers = 0
    if filter_layers:
        filter_string = '-Y2000'
    else: 
        filter_string = '-all'
    
    if 0: # run this to plot current vs firing freq
        alldata = pd.read_csv(join(csv_path,'compiled_firing_freq-FPC.csv'))
        alldata = alldata.loc[(alldata['region'] == brain_region) & (alldata['cell_type'] == cell_type)]

        if filter_layers:
            alldata = alldata[alldata[distvar] > threshold]

        # Function to remove outliers using the Z-score method for each group
        def remove_outliers(group):
            threshold = 10 # Z-score threshold, you can adjust this based on your data
            mean_firing_freq = np.mean(group['mean_firing_frequency'])
            std_firing_freq = np.std(group['mean_firing_frequency'])
            return group[abs(group['mean_firing_frequency'] - mean_firing_freq) < threshold * std_firing_freq]

        # Removing outliers from the 'pA/pF' column for each current injection group
        cleaned_df = alldata#.groupby('plot_pA/pF').apply(remove_outliers).reset_index(drop=True)

        # Print the DataFrame containing outliers
        outliers_df = alldata[~alldata.index.isin(cleaned_df.index)]
        print("Outliers firing:",len(outliers_df))

        xaxisstr = 'est_pA/pF'
        selectdata = cleaned_df.loc[(cleaned_df[xaxisstr] <= 30) & (cleaned_df[xaxisstr] >=0)]
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
        
        # # _____________ ANOVA statistics just sex and strain ____________________-
        # region_path = join(save_path,brain_region)
        # if not isdir(region_path):
        #     mkdir(region_path)

        # unique_pApF_values = selectdata[xaxisstr].unique().tolist()
        # for inj in unique_pApF_values: #get unique pApF and do the stats for each injection value
        #     current_inj_df = selectdata.loc[(selectdata[xaxisstr] == inj)]
        #     model = ols('mean_firing_frequency ~ C(strain)',data=current_inj_df).fit()
        #     result = sm.stats.anova_lm(model,typ=2)
        #     result.to_csv(join(save_path,brain_region,str(inj)+brain_region+'.csv'), index=True)

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

            nums = np.random.uniform(-1,1,len(hAPPdata['est_pA/pF']))
            s = plt.scatter(pd.to_numeric(hAPPdata['est_pA/pF'])+nums/2,
                            pd.to_numeric(hAPPdata['mean_firing_frequency']),
                            c=pd.to_numeric(hAPPdata[distvar]),
                            vmin=colormin,
                            vmax=colormax,
                            cmap=colormap)
            plt.title("hAPP " +brain_region)
            plt.xlabel("Current Injection (pA/pF)")
            plt.ylabel("Mean firing frequency (Hz)")
            plt.ylim([0,250])
            plt.colorbar(label=distvar)

            plt.figure()
            nums = np.random.uniform(-1,1,len(B6Jdata['est_pA/pF']))
            s = plt.scatter(pd.to_numeric(B6Jdata['est_pA/pF'])+nums/2,
                            pd.to_numeric(B6Jdata['mean_firing_frequency']),
                            c=pd.to_numeric(B6Jdata[distvar]),
                            vmin=colormin,
                            vmax=colormax,
                            cmap=colormap)
            plt.title("B6J "+brain_region)
            plt.xlabel("Current Injection (pA/pF)")
            plt.ylabel("Mean firing frequency (Hz)")
            plt.ylim([0,250])
            plt.colorbar(label=distvar)
            plt.show()
    
        fig, axs = plt.subplots()
        fig.set_size_inches(4,4)
        
        # Plotting the error bar plot
        lw = 2
        ms = 5

        plt.errorbar(current_injection_values_hAPP, mean_values_hAPP, yerr=sem_values_hAPP, color='royalblue',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        plt.errorbar(current_injection_values_B6J, mean_values_B6J, yerr=sem_values_B6J, color='k',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        plt.xlabel('Current Injection (pA/pF)')
        plt.ylabel('Mean Firing Frequency (Hz)')
        plt.ylim([0,250])
        plt.xlim([0,32])
        plt.legend(['hAPP','B6J'])

        try:
            plt.title(brain_region + ': hAPP ('+ str(hAPPdata.groupby(xaxisstr)['mean_firing_frequency'].size()[0]) +'), B6 (' + str(B6Jdata.groupby(xaxisstr)['mean_firing_frequency'].size()[0]) +')')
        except:
            print('no title')
        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region+cell_type+'firing.svg'),dpi=300,format='svg')

        if 0:
            # normality violin plots - one plot with B6 vs hAPP
            fig3, axs3 = plt.subplots(1)
            fig3.set_size_inches(16,8)

            sns.violinplot(data=selectdata,x='est_pA/pF',y='mean_firing_frequency',hue='strain',split=True,ax=axs3).set(title="hAPP vs B6J")
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
        alldata = alldata.loc[(alldata['MT-celltau'] > 0) & (alldata['MT-celltau'] < 85)& (alldata['MT-Rm'] >0)& (alldata['RMP'] >-80)]

        if filter_layers:
            alldata = alldata[alldata[distvar] > threshold]

        # Function to remove outliers using the Z-score method for each group
        def remove_outliers(group,metric):
            threshold = 10 # Z-score threshold, you can adjust this based on your data
            mean_firing_freq = np.mean(group[metric])
            std_firing_freq = np.std(group[metric])
            return group[abs(group[metric] - mean_firing_freq) < threshold * std_firing_freq]

        metrics = ['membrane_capacitance','MT-Rm','RMP','MT-celltau']

        for metric in metrics:
            cleaned_df = alldata.groupby('strain').apply(lambda x: remove_outliers(x, metric)).reset_index(drop=True)

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

            tau_list = np.array(pd.unique(fn_df['MT-celltau']))
            tau = np.median(tau_list)

            cap_list = np.array(pd.unique(fn_df['membrane_capacitance']))
            capacitance = np.median(cap_list)

            in_rest_list = np.array(pd.unique(fn_df['MT-Rm']))
            input_resistance = np.median(in_rest_list)

            rmp_list = np.array(pd.unique(fn_df['RMP']))
            rmp = np.median(rmp_list)

            avg_params.append([fn,strain,sex,hemisphere,tau,capacitance,input_resistance,rmp])

        avgdata = pd.DataFrame(avg_params, columns =['filename','strain','sex','hemisphere','membrane_tau','membrane_capacitance','input_resistance','RMP'])

        # print("hAPP :",len(avgdata[avgdata['strain']=='hAPPKI']))
        # print("B6 :",len(avgdata[avgdata['strain']=='B6']))

        # t test statistics
        pas_stats = []
        def unpairedTTest(avgdata,measured_metric):
            hAPP = avgdata[avgdata['strain']=='hAPPKI']
            B6J = avgdata[avgdata['strain']=='B6']

            group1 = hAPP[measured_metric]
            group2 = B6J[measured_metric]

            stat, pvalue = ttest_ind(group1,group2,equal_var = 0)
            nGroup1 = len(group1)
            nGroup2 = len(group2)
            
            # print('hAPP variance: ',statistics.variance(group1))
            # print('B6J variance: ',statistics.variance(group2))

            return [measured_metric,nGroup1,nGroup2,stat,pvalue]
        

        pas_stats.append(unpairedTTest(avgdata,'membrane_tau'))
        pas_stats.append(unpairedTTest(avgdata,'input_resistance'))
        pas_stats.append(unpairedTTest(avgdata,'membrane_capacitance'))
        pas_stats.append(unpairedTTest(avgdata,'RMP'))

        with open(join(save_path,"stats","pas_stats_"+brain_region+filter_string+cell_type+".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(['metric','n (hAPP)','n (B6J)','stat','p-value'])
            writer.writerows(pas_stats)


        # generate plots
        PROPS = setProps('black')
        fig2, axs2 = plt.subplots(ncols=4)
        fig2.set_size_inches(10,3)
        w = .2
        ms = 4
        huestr = "strain"
        palstr = ['k','royalblue']
        plot_order = ["B6", "hAPPKI"]

        def makeBoxplot(axs,data,metric,metric_str,axis_num,props):
            sns.boxplot(y=metric,x="strain",data=data,**props,width=w,ax=axs[axis_num], order = plot_order)
            sns.swarmplot(y=metric,x="strain",hue=huestr,data=data,zorder=.5,ax=axs[axis_num],palette=palstr,size=ms, order = plot_order)
            axs[axis_num].set(ylabel=metric_str,xlabel="")
            axs[axis_num].get_legend().remove()
            #axs[axis_num].set_ylim([-100,-20])

            # set up annotator for p values to be plotted automatically
            # https://levelup.gitconnected.com/statistics-on-seaborn-plots-with-statannotations-2bfce0394c00
            # https://github.com/trevismd/statannotations/tree/master

            pairs = [('B6','hAPPKI')]
            hue_plot_params = {'data': data, 'x': 'strain','y': metric,}
            annotator = Annotator(axs[axis_num], pairs, **hue_plot_params)
            pvalues = [.0024293857] # put actual values here.
            formatted_pvalues = [f'p={pvalue:.2g}' for pvalue in pvalues]
            annotator.set_custom_annotations(formatted_pvalues)
            annotator.annotate()
            
            # annotator.configure(test='Mann-Whitney').apply_and_annotate() # use this to run the stats in the plotting
            # annotator.configure(text_format="simple") # turn this on to show p <= 0.05 for instance

            return

        makeBoxplot(axs=axs2,data=avgdata,metric="RMP",metric_str="resting membrane potential (mV)",axis_num=0,props=PROPS)
        makeBoxplot(axs=axs2,data=avgdata,metric="membrane_tau",metric_str="tau (ms)",axis_num=1,props=PROPS)
        makeBoxplot(axs=axs2,data=avgdata,metric="input_resistance",metric_str="membrane resistance (M$\Omega$)",axis_num=2,props=PROPS)
        makeBoxplot(axs=axs2,data=avgdata,metric="membrane_capacitance",metric_str="membrane capacitance (pF)",axis_num=3,props=PROPS)

        plt.suptitle(brain_region + ': hAPP ('+ str(len(avgdata[avgdata['strain']=='hAPPKI'])) +'), B6 (' + str(len(avgdata[avgdata['strain']=='B6'])) +')')
        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region + filter_string + cell_type + '-pas.svg'),dpi=300,format='svg')


    if 1: # plot a boxplot for a spike params
        alldata = pd.read_csv(join(csv_path,'compiled_spike_params-FPC.csv'))
        
        if filter_layers:
            alldata = alldata[alldata[distvar] > threshold]

        alldata = alldata.loc[(alldata['APnum'] == 0) & (alldata['sweep'] == 10) & (alldata['cell_type'] == cell_type)& (alldata['region'] == brain_region)]
        # Function to remove outliers using the Z-score method for each group
        def remove_outliers(group,metric):
            threshold = 10  # Z-score threshold, you can adjust this based on your data
            mean_firing_freq = np.mean(group[metric])
            std_firing_freq = np.std(group[metric])
            return group[abs(group[metric] - mean_firing_freq) < threshold * std_firing_freq]

        metrics = ['AHP','threshold','dV/dt max','AP peak']

        for metric in metrics:
            # Removing outliers from the 'pA/pF' column for each current injection group
            cleaned_df = alldata.groupby('strain').apply(lambda x: remove_outliers(x, metric)).reset_index(drop=True)

        # Print the DataFrame containing outliers
        outliers_df = alldata[~alldata.index.isin(cleaned_df.index)]
        print("Outliers spike:",len(outliers_df))
        
        PROPS = setProps('black')
        fig3, axs3 = plt.subplots(ncols=5)#,nrows=2)
        fig3.set_size_inches(10,3)
        w = .2

        huestr = "strain"
        palstr = ['k','royalblue']
        plot_order = ["B6", "hAPPKI"]

        sns.boxplot(y="AP peak",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs3[0],order = plot_order)
        sns.swarmplot(y="AP peak",x="strain",hue=huestr,data=cleaned_df,zorder=.5,ax=axs3[0],palette=palstr,size=ms,order = plot_order)
        axs3[0].set(ylabel="AP peak (mV)",xlabel="")
        axs3[0].set_ylim([20,80])

        sns.boxplot(y="AP hwdt",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs3[1],order = plot_order)
        sns.swarmplot(y="AP hwdt",x="strain",hue=huestr,data=cleaned_df,zorder=.5,ax=axs3[1],palette=palstr,size=ms,order = plot_order)
        axs3[1].set(ylabel="AP hwdt (ms)",xlabel="")
        axs3[1].set_ylim([.2,1.1])

        sns.boxplot(y="AHP",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs3[4],order = plot_order)
        sns.swarmplot(y="AHP",x="strain",data=cleaned_df,zorder=.5,ax=axs3[4],palette=palstr,size=ms,order = plot_order)
        axs3[4].set(ylabel="AHP",xlabel="")

        sns.boxplot(y="threshold",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs3[3],order = plot_order)
        sns.swarmplot(y="threshold",x="strain",hue=huestr,data=cleaned_df,zorder=.5,ax=axs3[3],palette=palstr,size=ms,order = plot_order)
        axs3[3].set(ylabel="threshold (mV)",xlabel="")
        axs3[3].set_ylim([-60,-10])

        sns.boxplot(y="dV/dt max",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs3[2],order = plot_order)
        sns.swarmplot(y="dV/dt max",x="strain",hue=huestr,data=cleaned_df,zorder=.5,ax=axs3[2],palette=palstr,order = plot_order)
        axs3[2].set(ylabel="dV/dt max (mV/s)",xlabel="")
        axs3[2].set_ylim([100,800])

        for i in [0,1,2,3]:
            axs3[i].get_legend().remove()
            # print(0)
        # handles, labels = axs2[3].get_legend_handles_labels()
        # fig2.legend(handles, labels, loc='upper right')

        plt.suptitle(brain_region + ': hAPP ('+ str(len(cleaned_df[cleaned_df['strain']=='hAPPKI'])) +'), B6 (' + str(len(cleaned_df[cleaned_df['strain']=='B6'])) +')')
        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region+filter_string+cell_type+'-spike.svg'),dpi=300,format='svg')
         # ________ Test for ttest _________ 
        pas_stats = []
        def unpairedTTest(avgdata,measured_metric):
            hAPP = avgdata[avgdata['strain']=='hAPPKI']
            B6J = avgdata[avgdata['strain']=='B6']

            group1 = hAPP[measured_metric]
            group2 = B6J[measured_metric]

            stat, pvalue = ttest_ind(group1,group2,equal_var = 0)
            nGroup1 = len(group1)
            nGroup2 = len(group2)
            
            return [measured_metric,nGroup1,nGroup2,stat,pvalue]
        

        pas_stats.append(unpairedTTest(alldata,'AP peak'))
        pas_stats.append(unpairedTTest(alldata,'AP hwdt'))
        pas_stats.append(unpairedTTest(alldata,'AHP'))
        pas_stats.append(unpairedTTest(alldata,'threshold'))
        pas_stats.append(unpairedTTest(alldata,'dV/dt max'))

        with open(join(save_path,"stats","spike_stats_"+brain_region+filter_string+cell_type+".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(['metric','n (hAPP)','n (B6J)','stat','p-value'])
            writer.writerows(pas_stats)

    if 0: # run this to plot ap peak vs ap num, not yet done 
        alldata = pd.read_csv(join(csv_path,'compiled_spike_params-FPC.csv'))
        alldata = alldata.loc[(alldata['region'] == brain_region) & (alldata['cell_type'] == cell_type)]

        # Removing outliers from the 'pA/pF' column for each current injection group
        cleaned_df = alldata#.groupby('APnum').apply(remove_outliers).reset_index(drop=True)
        
        xaxisstr = 'APnum'
        yaxisstr = 'AP peak'

        selectdata = cleaned_df.loc[(cleaned_df[xaxisstr] <= 30) & (cleaned_df[xaxisstr] >=0) & (cleaned_df['pA/pF'] == 10)]
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

        # inj1 = APnums[APnums['pA/pF'] == 4]
        # inj2 = APnums[APnums['pA/pF'] == 20]
        # inj3 = APnums[APnums['pA/pF'] == 30]
        # inj_list = [inj1,inj2,inj3]

        metric = "AP peak"
        PROPS = setProps('black')
        # Plotting the error bar plot
        lw = 2
        ms = 5
        
        plt.errorbar(current_injection_values_hAPP, mean_values_hAPP, yerr=sem_values_hAPP, color='royalblue',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        plt.errorbar(current_injection_values_B6J, mean_values_B6J, yerr=sem_values_B6J, color='k',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        
        plt.xlabel('AP Number')
        plt.ylabel('AP Peak (mV)')

        plt.title(brain_region)
        
        # plt.ylim([0,40])
        # plt.xlim([0,32])
        plt.legend(['hAPP','B6J'])

        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region+cell_type+'APpeak.svg'),dpi=300,format='svg')