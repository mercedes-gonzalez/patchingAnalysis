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
    
    brain_region = 'mPFC'
    distvar = 'Y'

    save_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Figs/Fall 2023/'
    if 1: # run this to plot current vs firing freq
        alldata = pd.read_csv(join(csv_path,'compiled_firing_freq-FPC.csv'))
        alldata = alldata.loc[(alldata['region'] == brain_region) & (alldata['cell_type'] == 'interneuron')]
        
        # Function to remove outliers using the Z-score method for each group
        def remove_outliers(group):
            threshold = 3# Z-score threshold, you can adjust this based on your data
            mean_firing_freq = np.mean(group['mean_firing_frequency'])
            std_firing_freq = np.std(group['mean_firing_frequency'])
            return group[abs(group['mean_firing_frequency'] - mean_firing_freq) < threshold * std_firing_freq]

        # Removing outliers from the 'pA/pF' column for each current injection group
        cleaned_df = alldata#.groupby('plot_pA/pF').apply(remove_outliers).reset_index(drop=True)
        print('cleaned: ',len(cleaned_df))

        # Print the DataFrame containing outliers
        outliers_df = alldata[~alldata.index.isin(cleaned_df.index)]
        print("Outliers:",len(outliers_df))

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

        # ___________________ between all groups stats _________________________
        # Test groups for normality first.
        stats = []
        grouped = selectdata.groupby([xaxisstr, 'strain'])
        for group_name, group_data in grouped['mean_firing_frequency']:
            shapiro_stat, shapiro_p_value = shapiro(group_data)
            nsamples = len(group_data)
            print(group_name)
            print(f"Group: {group_name}, N: {nsamples}, Shapiro-Wilk Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p_value:.4f}")
            stats.append([group_name[0],group_name[1],nsamples,shapiro_stat,shapiro_p_value])
        
        with open("/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Figs/Fall 2023/pas_params_stats_"+brain_region+".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow([xaxisstr,'strain','numSamples','SW-stat','p-value'])
            writer.writerows(stats)
        
        # _____________ ANOVA statistics just sex and strain ____________________-
        region_path = join(save_path,brain_region)
        if not isdir(region_path):
            mkdir(region_path)

        save_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Figs/Fall 2023/'
        unique_pApF_values = selectdata[xaxisstr].unique().tolist()
        for inj in unique_pApF_values: #get unique pApF and do the stats for each injection value
            current_inj_df = selectdata.loc[(selectdata[xaxisstr] == inj)]
            model = ols('mean_firing_frequency ~ C(strain)',data=current_inj_df).fit()
            result = sm.stats.anova_lm(model,typ=2)
            print(result)
            result.to_csv(join(save_path,brain_region,str(inj)+brain_region+'.csv'), index=True)

        # ___________________ hAPP vs B6 stats _________________________
        stats = []
        grouped = selectdata.groupby(['strain',xaxisstr])
        for group_name, group_data in grouped['mean_firing_frequency']:
            shapiro_stat, shapiro_p_value = shapiro(group_data)
            nsamples = len(group_data)
            print(f"Group: {group_name}, N: {nsamples}, Shapiro-Wilk Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p_value:.4f}")
            stats.append([group_name[0],group_name[1],nsamples,shapiro_stat,shapiro_p_value])
        
        with open(join(region_path,"firing_stats-"+brain_region+"-B6vshAPP.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(['strain','current_inj','numSamples','SW-stat','p-value'])
            writer.writerows(stats)
        # _______________________________________________________________
        
        if 1:
            hAPPdata = hAPPdata[hAPPdata[distvar] != 'na']
            B6Jdata = B6Jdata[B6Jdata[distvar] != 'na']
            plt.figure()
            colormin=0
            colormax = max(pd.to_numeric(hAPPdata[distvar]))
            print(colormax)
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
            plt.ylim([0,300])
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
            plt.ylim([0,300])
            plt.colorbar(label=distvar)
            plt.show()
    
        fig, axs = plt.subplots()
        fig.set_size_inches(4,4)
        
        # Plotting the error bar plot
        lw = 2
        ms = 5

        plt.errorbar(current_injection_values_hAPP, mean_values_hAPP, yerr=sem_values_hAPP, color='k',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        plt.errorbar(current_injection_values_B6J, mean_values_B6J, yerr=sem_values_B6J, color='royalblue',fmt='o', markeredgewidth=lw,linewidth=lw,capsize=5,markersize=ms,markerfacecolor='white')
        plt.xlabel('Current Injection (pA/pF)')
        plt.ylabel('Mean Firing Frequency (Hz)')
        plt.ylim([0,250])
        plt.xlim([0,32])
        plt.legend(['hAPP','B6J'])
        plt.title(brain_region)
        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region+'firing.svg'),dpi=300,format='svg')
        plt.show()

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
            B6data = selectdata[selectdata['strain'] == 'B6J']

            sns.violinplot(data=hAPPdata,x='pApF',y='mean_firing_frequency',hue='sex',split=True,ax=axs3[0]).set(title="hAPP")
            sns.violinplot(data=B6data,x='pApF',y='mean_firing_frequency',hue='sex',split=True,ax=axs3[1]).set(title="B6J")

            plt.tight_layout()


        plt.show()
    if 0: # plot a boxplot for a passive param
        alldata = pd.read_csv(join(csv_path,'compiled_pas_params-FPC.csv'))
        alldata = alldata.loc[(alldata['region'] == brain_region) & (alldata['cell_type'] == 'interneuron')]
        alldata = alldata.loc[(alldata['MT-celltau'] > 0) & (alldata['MT-celltau'] < 85)& (alldata['MT-Rm'] >0)& (alldata['RMP'] >-80)]
        
        # Function to remove outliers using the Z-score method for each group
        def remove_outliers(group,metric):
            threshold = 3 # Z-score threshold, you can adjust this based on your data
            mean_firing_freq = np.mean(group[metric])
            std_firing_freq = np.std(group[metric])
            return group[abs(group[metric] - mean_firing_freq) < threshold * std_firing_freq]

        metrics = ['membrane_capacitance','MT-Rm','RMP','MT-celltau']

        for metric in metrics:
            cleaned_df = alldata.groupby('strain').apply(lambda x: remove_outliers(x, metric)).reset_index(drop=True)
        print("OUTLIERS: ",len(alldata)-len(cleaned_df))

        avg_params = []
        # average the params
        list_fn = pd.unique(cleaned_df['filename'])

        for num,fn in enumerate(list_fn):
            fn_df = cleaned_df.loc[cleaned_df['filename'] == fn]
            strain = fn_df.iat[0,4] # change these if add more columns
            sex = fn_df.iat[0,3]
            hemisphere = fn_df.iat[0,2]

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
        print("hAPP :",len(avgdata[avgdata['strain']=='hAPPKI']))
        print("B6 :",len(avgdata[avgdata['strain']=='B6']))

        PROPS = setProps('black')
        fig2, axs2 = plt.subplots(ncols=4)
        fig2.set_size_inches(10,3)
        w = .2
        ms = 6
        huestr = "strain"
        palstr = ['k','royalblue']
        plot_order = ["B6", "hAPPKI"]

        sns.boxplot(y="RMP",x="strain",data=avgdata,**PROPS,width=w,ax=axs2[3], order = plot_order)
        sns.swarmplot(y="RMP",x="strain",hue=huestr,data=avgdata,zorder=.5,ax=axs2[3],palette=palstr,size=ms, order = plot_order)
        axs2[3].set(ylabel="resting membrane potential (mV)",xlabel="")
        axs2[3].set_ylim([-100,-20])

        sns.boxplot(y="membrane_tau",x="strain",data=avgdata,**PROPS,width=w,ax=axs2[2], order = plot_order)
        sns.swarmplot(y="membrane_tau",x="strain",hue=huestr,data=avgdata,zorder=.5,ax=axs2[2],palette=palstr,size=ms, order = plot_order)
        axs2[2].set(ylabel="tau (ms)",xlabel="")
        axs2[2].set_ylim([0,50])

        sns.boxplot(y="input_resistance",x="strain",data=avgdata,**PROPS,width=w,ax=axs2[0], order = plot_order)
        sns.swarmplot(y="input_resistance",x="strain",hue=huestr,data=avgdata,zorder=.5,ax=axs2[0],palette=palstr,size=ms, order = plot_order)
        axs2[0].set(ylabel="membrane resistance (M$\Omega$)",xlabel="")
        axs2[0].set_ylim([0,500])

        sns.boxplot(y="membrane_capacitance",x="strain",data=avgdata,**PROPS,width=w,ax=axs2[1], order = plot_order)
        sns.swarmplot(y="membrane_capacitance",x="strain",hue=huestr,data=avgdata,zorder=.5,ax=axs2[1],palette=palstr,size=ms, order = plot_order)
        axs2[1].set(ylabel="membrane capacitance (pF)",xlabel="")
        axs2[1].set_ylim([0,100])

        for i in [0,1,2,3]:
            axs2[i].get_legend().remove()

        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region + '-pas.svg'),dpi=300,format='svg')

        # pas_stats = []
        # def unpairedTTest(avgdata,measured_metric):
        #     hAPP = avgdata[avgdata['strain']=='hAPPKI']
        #     B6J = avgdata[avgdata['strain']=='B6']

        #     group1 = hAPP[measured_metric]
        #     group2 = B6J[measured_metric]

        #     stat, pvalue = ttest_ind(group1,group2,equal_var = 0)
        #     nGroup1 = len(group1)
        #     nGroup2 = len(group2)
            
        #     print(measured_metric)
        #     # print('hAPP variance: ',statistics.variance(group1))
        #     # print('B6J variance: ',statistics.variance(group2))

        #     return [measured_metric,nGroup1,nGroup2,stat,pvalue]
        

        # pas_stats.append(unpairedTTest(avgdata,'membrane_tau'))
        # pas_stats.append(unpairedTTest(avgdata,'input_resistance'))
        # pas_stats.append(unpairedTTest(avgdata,'membrane_capacitance'))
        # pas_stats.append(unpairedTTest(avgdata,'RMP'))

        # with open("/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Figs/Fall 2023/pas_stats_V1.csv", "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['metric','n (hAPP)','n (B6J)','stat','p-value'])
        #     writer.writerows(pas_stats)

    if 0: # plot a boxplot for a spike params
        plt.clf()
        alldata = pd.read_csv(join(csv_path,'compiled_spike_params-FPC.csv'))
        alldata = alldata.loc[(alldata['APnum'] == 0) & (alldata['sweep'] == 3) & (alldata['cell_type'] == 'interneuron')& (alldata['region'] == brain_region)]
        
        # Function to remove outliers using the Z-score method for each group
        def remove_outliers(group,metric):
            threshold = 2.5  # Z-score threshold, you can adjust this based on your data
            mean_firing_freq = np.mean(group[metric])
            std_firing_freq = np.std(group[metric])
            return group[abs(group[metric] - mean_firing_freq) < threshold * std_firing_freq]

        metrics = ['AHP','threshold','dV/dt max','AP peak']

        for metric in metrics:
            # Removing outliers from the 'pA/pF' column for each current injection group
            cleaned_df = alldata.groupby('strain').apply(lambda x: remove_outliers(x, metric)).reset_index(drop=True)
        print("data: ",len(cleaned_df))

        PROPS = setProps('black')
        fig2, axs2 = plt.subplots(ncols=5)#,nrows=2)
        fig2.set_size_inches(10,3)
        w = .2
        huestr = "strain"
        palstr = ['k','royalblue']
        sns.boxplot(y="AP peak",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[0])
        sns.swarmplot(y="AP peak",x="strain",hue=huestr,data=cleaned_df,zorder=.5,ax=axs2[0],palette=palstr)
        axs2[0].set(ylabel="AP peak (mV)",xlabel="")
        axs2[0].set_ylim([0,60])

        sns.boxplot(y="AP hwdt",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[1])
        sns.swarmplot(y="AP hwdt",x="strain",hue=huestr,data=cleaned_df,zorder=.5,ax=axs2[1],palette=palstr)
        axs2[1].set(ylabel="AP hwdt (ms)",xlabel="")
        axs2[1].set_ylim([.2,1.1])

        sns.boxplot(y="AHP",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[4])
        sns.swarmplot(y="AHP",x="strain",data=cleaned_df,zorder=.5,ax=axs2[4],palette=palstr)
        axs2[4].set(ylabel="AHP",xlabel="")

        sns.boxplot(y="threshold",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[3])
        sns.swarmplot(y="threshold",x="strain",hue=huestr,data=cleaned_df,zorder=.5,ax=axs2[3],palette=palstr)
        axs2[3].set(ylabel="threshold (mV)",xlabel="")
        axs2[3].set_ylim([-60,-10])

        sns.boxplot(y="dV/dt max",x="strain",data=cleaned_df,**PROPS,width=w,ax=axs2[2])
        sns.swarmplot(y="dV/dt max",x="strain",hue=huestr,data=cleaned_df,zorder=.5,ax=axs2[2],palette=palstr)
        axs2[2].set(ylabel="dV/dt max (mV/s)",xlabel="")
        axs2[2].set_ylim([100,800])

        for i in [0,1,2,3]:
            axs2[i].get_legend().remove()
            # print(0)
        # handles, labels = axs2[3].get_legend_handles_labels()
        # fig2.legend(handles, labels, loc='upper right')

        plt.tight_layout()
        plt.savefig(join(save_path,'svgs',brain_region+'-spike.svg'),dpi=300,format='svg')
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
            
            print(measured_metric)

            return [measured_metric,nGroup1,nGroup2,stat,pvalue]
        

        pas_stats.append(unpairedTTest(alldata,'AP peak'))
        pas_stats.append(unpairedTTest(alldata,'AP hwdt'))
        pas_stats.append(unpairedTTest(alldata,'AHP'))
        pas_stats.append(unpairedTTest(alldata,'threshold'))
        pas_stats.append(unpairedTTest(alldata,'dV/dt max'))

        with open("/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Figs/Fall 2023/spike_stats_"+brain_region+".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(['metric','n (hAPP)','n (B6J)','stat','p-value'])
            writer.writerows(pas_stats)


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
