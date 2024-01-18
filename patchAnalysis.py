'''
All of the functions and extra code needed for patch analysis script.

Mercedes Gonzalez. August 2023. 
mercedesmg.com
Precision Biosystems Lab | Georgia Institute of Technology
Version Control: https://github.com/mercedes-gonzalez/patchingAnalysis
Based on: https://github.com/ViktorJOlah/Firing_pattern_analysis
'''

from os.path import join, exists
import numpy as np
import scipy.signal as sig
import pyabf
import matplotlib.pyplot as plt
import os
from scipy import optimize, integrate
import regex as re

def getResponseDataSweep(d,sweepNum):
    return d.response[sweepNum,:]

def getCommandSweep(d,sweepNum):
    return d.command[sweepNum,:]

def open_myabf(abf_name):
    return pyabf.ABF(abf_name)

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

def calc_pas_params(d,filename,base_fn): # filename is the image path, base_fn is the name of the abf
    # initialize the array to save all the parameters
    n_sweeps = d.numSweeps
    n_params = 5
    all_data = np.empty((n_sweeps, n_params))

    # for each sweep in the abf, find the passive properties and save to the array 
    for sweep in range(n_sweeps): 
        voltage_data = getResponseDataSweep(d,sweep)
        dt = 1/d.sampleRate
        command_current = getCommandSweep(d,sweep)
        del_com = np.diff(command_current)
        starts = np.where(del_com<0)
        ends = np.where(del_com>0)
        # these should be for passive properties (ie 1st step down)
        const = 0
        passive_start = starts[0][0] + const
        passive_end = ends[0][0] - const

        mean1 = np.mean(voltage_data[0 : passive_start-1])  #calculate Rm/input_resistance
        mean2 = np.mean(voltage_data[int(passive_start + (0.1 / dt)) : passive_end])

        holding = np.mean(command_current[0: passive_start-10])
        pas_stim = np.mean(command_current[passive_start + 10 : passive_start + 110]) - holding

        input_resistance = (abs(mean1-mean2) / abs(pas_stim) ) * 1000 # Steady state delta V/delta I
        resting = mean1 - (input_resistance * holding) / 1000

        import scipy.optimize

        X1 = d.time[passive_start : int((passive_start + (0.1 / dt)))]           #calculate membrane tau
        Y1 = voltage_data[passive_start : int((passive_start + (0.1 / dt)))]

        p0 = (20, 10, voltage_data[passive_end])

        try:
            npoints = 25
            params, cv = scipy.optimize.curve_fit(monoExp, X1[::npoints], Y1[::npoints], p0, maxfev = 100000)
            m, t, b = params
            sampleRate = int(1 / dt / 1000)+1
            membrane_tau =  ((1 / t) / sampleRate) * 1e6 / abs(pas_stim)
            membrane_capacitance = membrane_tau / input_resistance *1000
        except:
            m = .5
            t = .5
            b = .5
            membrane_tau = .5
            membrane_capacitance = .5
            print('failed to fit.')

        """
            if tauSec > 100 or tauSec < 6:
                membrane_tau = 'nan'
            else:
                membrane_tau = tauSec

            if Rm > 10000 or Rm < 40 or membrane_tau == 'nan':
                Input_resistance = 'nan'
            else:
                Input_resistance = Rm
            if cap > 300 or cap < 5 or Input_resistance == 'nan':
                membrane_capacitance = 'nan'
            else:
                membrane_capacitance = cap
        except:
            pass
        """

        # find error in fit
        fit_err = np.average(abs((monoExp(d.time[passive_start:passive_end],m,t,b)-voltage_data[passive_start:passive_end])))

        if 1:
            # find limits for the y-axis
            max_lim = np.max(voltage_data[passive_start:passive_end]) + 5
            min_lim = np.min(voltage_data[passive_start:passive_end]) - 5
            
            # plot for checking fitting
            plt.plot(d.time,voltage_data)
            plt.plot(X1,Y1)
            plt.plot(d.time,monoExp(d.time, m, t, b))
            plt.scatter([d.time[passive_start],d.time[passive_end]],[voltage_data[passive_start],voltage_data[passive_end]],c='red')
            plt.ylim([min_lim,max_lim])
            # plt.xlim([.4,1.1])
            plt.title(sweep)
            if sweep == 0:
                plt.savefig(filename+".png")
                # plt.show()
            plt.clf()

        all_data[sweep,:] = [membrane_tau, input_resistance, membrane_capacitance, resting, fit_err]
    return all_data 

def running_mean(x, N):                                                     #running mean to avoid measuring noise
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def moving_average(x, w): # for memtest 
    return np.convolve(x, np.ones(w), 'same') / w

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def analyzeMemtest(myData,base_fn,save_path,verbose):
    time = myData.time
    rec_current = myData.response

    expFunc = lambda t,m,tau,I1 : m*np.exp(-t/tau)+I1

    pico = 10**-12
    milli = 10**-3
    mega = 10**-6
    # rec_current = rec_current* # convert current from pA to A

    filtered_current = moving_average(rec_current,20) # A

    # define indices
    max_current = rec_current.max() #A
    start_idx = 71 #np.argmax(filtered_current)+10 # hardcoded indices bc i'm tired
    end_idx = start_idx + 300

    # parse data 
    T_fit = time[start_idx:end_idx] # sec
    I_fit = filtered_current[start_idx:end_idx] # A

    # remove steady state parts? 
    I1 = np.average(rec_current[600:700]) # hardcoded indices
    try:
        [m, tau, Iss], _ = optimize.curve_fit(expFunc, T_fit, I_fit,p0=(600,.0035,30),maxfev = 100000) #bounds=((0,3000),(0,0.0075),(-200,200))
        
    except:
        print(base_fn, " could not find fit.")
        return
    
    # print('\tmetrics: ', [m, tau, Iss])
    Iprev = np.average(rec_current[0:10])
    Idss = (I1-Iprev)
    Ipeak = m + Iss
    Id = Ipeak-Iprev

    deltaV = 10*milli # convert from mV to V
    Ra = abs(deltaV/(Id)) # Ohms
    Rm = (deltaV - (Ra*Idss))/Idss # Ohms
    R_tot = 1/((1/Ra)+(1/Rm)) #Ohms
    Cm = (tau)/R_tot #Farads
    cell_tau = Rm*Cm

    if verbose:
        print("I1 = ",I1)
        print("Iprev = ", Iprev)
        print("Idss =", Idss)
        print("Ipeak = ",Ipeak)
        print("Id = ",Id)
        print("Iss = ",Iss)
        print("Ra (MOhm)= ",Ra/mega)
        print("cell tau (ms) = ",Cm*Rm)
        print("Rm (MOhm)= ", Rm/mega)
        print("Rt (MOhm)= ", R_tot/mega)
        print("Cm (pF)= ", Cm)

    # plot and save for checking. also calculate error in fitting. 
    fit_err = np.average(abs((expFunc(time[start_idx:end_idx],m,tau,I1)-rec_current[start_idx:end_idx])))

    # find limits for the y-axis
    max_lim = np.max(rec_current[start_idx:end_idx]) + 100 
    min_lim = np.min(rec_current[start_idx:end_idx]) - 100

    # plot for checking fitting
    plt.plot(time,rec_current)
    plt.plot(T_fit,expFunc(T_fit, m, tau, Iss))
    plt.plot(T_fit,I_fit)
    plt.scatter([time[start_idx],time[end_idx]],[rec_current[start_idx],rec_current[end_idx]],c='red')
    
    plt.ylim([min_lim,max_lim])
    # plt.xlim([0,.05])
    plt.savefig(join(save_path,base_fn+'.png'))
    plt.clf()

    # return the stats we need 
    access_resistance = Ra
    holding_current = Iss
    membrane_resistance = Rm
    membrane_capacitance = Cm
    fit_error = fit_err

    # print(base_fn,'\t',access_resistance/mega, '\t',holding_current, '\t',membrane_resistance/mega, '\t',membrane_capacitance)
    return [access_resistance/mega, holding_current, membrane_resistance/mega, membrane_capacitance, cell_tau/milli, fit_error]
    # return [access_resistance/mega, fit_error]

''' analyze memtest
def analyze_memtest(time,command_voltage,rec_current,fn,verbose):
    base_path = "/Users/mercedesgonzalez/Dropbox (GaTech)/patcherbotData/summary-stats/check-plots/memtests/"    
    expFunc = lambda t,A,tau,I1 : A*np.exp(-t/tau)+I1

    pico = 10**-12
    milli = 10**-3
    mega = 10**-6
    
    filtered_current = moving_average(rec_current,30)

    # define indices
    max_current = rec_current.max()
    # start_idx = np.abs(filtered_current - .8*max_current).argmin()
    # end_idx = np.abs(filtered_current - .2*max_current).argmin()
    start_idx = 5 # hardcoded indices bc i'm tired
    end_idx = 200 
    
    print("Max current: ",max_current)
    print("Start idx: ", start_idx)
    print("End idx: ", end_idx)

    # parse data 
    T_fit = time[start_idx:end_idx]
    I_fit = filtered_current[start_idx:end_idx]

    print(T_fit)
    print(I_fit)

    # remove steady state parts? 
    mid = int(len(rec_current)/2)
    quart = int(mid/2)
    T1 = time[quart:mid]
    I1 = np.average(rec_current[quart:mid])


    [m1, tau1, b1], _ = optimize.curve_fit(expFunc, T_fit, I_fit,p0=(600*pico,2,-100*pico),method='lm',sigma=1/(I_fit))
    
    Id = max_current-I1
    # # p0=(10,.003,-2)
    # Iss = (I1 + I2) / 2
    # deltaI = I1 - I2
    # deltaV = 10*milli
    # Rt = abs(deltaV/deltaI)

    # Q1,errorinQ = integrate.quad(func=expFunc,a=filtered_current[0],b=filtered_current[mid],args=(m1,tau1,b1))
    # Q2 = deltaI * tau1
    # Qt = Q1 + Q2 
    # Ra = tau1*deltaV/Qt
    # Rm = Rt - Ra
    # Cm = Qt*Rt/(deltaV*Rm)

    if verbose:
        print("I1 = ",I1/pico)
        print("I2 = ",I2/pico)      
        print("Ra (MOhm)= ",Ra*mega)
        print("tau (ms) = ",tau1/milli)
        print("deltaI (pA) = ",deltaI/pico)
        print("Iss (holding) = ",Iss/pico)
        print("Rm (MOhm)= ", Rm*mega)
        print("Rt (MOhm)= ", Rt*mega)
        print("Cm (pF)= ", Cm/pico)

    # plot and save for checking. also calculate error in fitting. 
    fit_err = np.average(abs((monoExp(time[start_idx:end_idx],m1,tau1,b1)-rec_current[start_idx:end_idx])))

    # find limits for the y-axis
    max_lim = np.max(rec_current[start_idx:end_idx]) + 5
    min_lim = np.min(rec_current[start_idx:end_idx]) - 5
    
    step1 = re.sub("/","",fn)
    step2 = re.sub("UsersmercedesgonzalezDropbox \(GaTech\)patcherbotData","",step1)
    step3 = re.sub("lvm","png",step2)
    fullfilename = join(base_path,step3)
    # plot for checking fitting
    plt.plot(time,rec_current)
    # plt.plot(X1,Y1)
    # plt.plot(time,monoExp(time, m1, tau1, b1))
    # plt.scatter([time[start_idx],time[end_idx]],[rec_current[start_idx],rec_current[end_idx]],c='red')
    # plt.ylim([min_lim,max_lim])
    # plt.xlim([0,.3])
    plt.savefig(fullfilename)
    plt.clf()

    # return the stats we need 
    access_resistance = Ra
    holding_current = Iss
    membrane_resistance = Rm
    membrane_capacitance = Cm
    fit_error = 0


    return [access_resistance, holding_current, membrane_resistance, membrane_capacitance, fit_error]
'''
''' calc sag
def calc_sag():
    global sag_amplitude
    global sag_timing


    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = 1/d.sampleRate
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    abf.setSweep(int(get_value("Sweep for sag parameters")), int(get_value("Voltage channel")))

    moving_ave = running_mean(current, 1000)

    sag_min = np.min(moving_ave[stim_start : int((stim_start + (0.15 / dt)))]) - np.mean(current[int(stim_start + (float(lenght1)*0.8 / dt)) : stim_end])
    sag_timing1 = (np.argmin(moving_ave[stim_start : int((stim_start + (0.15 / dt)))])) * dt

    sag_amplitude = sag_min
    sag_timing = sag_timing1
'''

def calc_freq(d,fn,save_path):     #calculate mean and max firing frequency
    # fn = int(fn)
    # d is a data object (custom defined class called data)
    dt = 1/d.sampleRate

    try: 
        del_com = np.diff(getCommandSweep(d,0))
    except:
        return
    
    starts = np.where(del_com<0)
    ends = np.where(del_com>0)
    stim_start = starts[0][1] # indices
    stim_end = ends[0][1] # indices

    # print('stim start and end:')
    # print(stim_start)
    # print(stim_end)

    stim_length = (stim_end - stim_start)/d.sampleRate

    # print("stimlength:", stim_length)
    all_avgs = np.empty((d.numSweeps,3))

    # make logical mask to determine current injection sweep
    command = getCommandSweep(d,0)
    baseline_cmd = np.array(command[0:10]).mean() #get baseline command (no input)
    is_on = command < baseline_cmd # Create logical mask for when command input is on

    num_rows = 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 10))
    
    def plot_subplots(ax,row,col,x,y,peakx,peaky,title,properties):
        # rowlab method only: 
        peaky = np.ones_like(peaky)*60

        ax[row,col].scatter(peakx,peaky+5,marker="|",color='r',s=20)
        ax[row,col].plot(x,y,linewidth=1)
        ax[row,col].set_title(title)

    positions = np.reshape(np.arange(num_cols*num_rows),(num_rows,num_cols))

    for sweep in range(d.numSweeps):
        current = getResponseDataSweep(d,sweep)
        command = getCommandSweep(d,sweep)
        AP_count = 0
        AP_list = []
        peaks = []

        # GET COMMAND
        baseline_cmd = np.array(command[0:10]).mean() #get baseline command (no input)
        input_dur = np.sum(is_on) # number of samples collected with input ON
        input_cmd = np.array(command[is_on]).mean() # get average of input command
        
        # GET NUMBER OF ACTION POTENTIALS
        baseline_data = np.array(current[0:10]).mean() # get baseline data
        current_findpeak = current[stim_start:stim_end+500] # make sure to get the edges
        time_findpeak = d.time[stim_start:stim_end+500]
        peaks, prop = sig.find_peaks(moving_average(current_findpeak,50),prominence=(10,None),height=(.2*(max(current)-min(current))+min(current)),width=(None,750)) # new method
        # peaks, prop = sig.find_peaks(moving_average(current_findpeak,50),prominence=.5,height=(.3*(max(current)-min(current))+min(current))) # sfn find peaks method

        currentinj = command[stim_start+5]

        # rowlab version
        # prop = None
        # for i in range(len(current_findpeak)):
        #     if current_findpeak[i] > -20:
        #         if len(AP_list) > 0 and abs(AP_list[-1] - i) > 300: 
        #             AP_count += 1
        #             AP_list.append(i)
        #         if len(AP_list) == 0:
        #             AP_count += 1
        #             AP_list.append(i)
        # peaks = AP_list
        
        num_APs = len(peaks)

        # add to the subplots
        if sweep > 1 and sweep < num_rows*num_cols:
            ax_idx = np.hstack(np.where(positions == sweep-2))
            row = ax_idx[0]
            col = ax_idx[1]
            plot_subplots(axes,row,col,time_findpeak,current_findpeak,time_findpeak[peaks],current_findpeak[peaks],currentinj,prop)

        # print("NUM APS:",num_APs)
        mean_freq = num_APs / float(stim_length)

        # max_freq = 1/((AP_list[1] - AP_list[0]) * dt)

        mean_firing_frequency = mean_freq
        # max_firing_frequency = max_freq
        all_avgs[sweep,:] = (sweep+1,currentinj,mean_firing_frequency)
    
    plt.savefig(join(save_path,str(fn)+'.png'))
    plt.title(fn)
    plt.clf()
    plt.close()
    # plt.show()
    return all_avgs

''' calc acc ratio
def calc_acc_ratio():

    global accommodation_ratio

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = 1/d.sampleRate
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))


    abf.setSweep(int(get_value("Sweep for accommodation ratio")), int(get_value("Voltage channel")))
                                               #calculate accommodation ratio

    AP_list = []

    for i in range(stim_start, stim_end):
        if current[i] > -20:
            if len(AP_list) > 0 and abs(AP_list[-1] - i) > 300: 
                AP_list.append(i)
            if len(AP_list) == 0:
                AP_list.append(i)

    accommodation_ratio1 = (AP_list[-1] - AP_list[-2]) / (AP_list[1] - AP_list[0])

    accommodation_ratio = accommodation_ratio1
'''
''' spike scaling
def spike_scaling():
    global frequency_list
    global current_list

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = 1/d.sampleRate

    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    sweep = 0
    freq_list = []
    curr_list =[]
    for i in range(abf.sweepCount):
        sweep = i
        abf.setSweep(sweep, int(get_value("Voltage channel")))

        AP_list = []

        for k in range(stim_start, stim_end):
            if current[k] > -20:
                if len(AP_list) > 0 and abs(AP_list[-1] - k) > 300: 
                    AP_list.append(k)
                if len(AP_list) == 0:
                    AP_list.append(k)

        if len(AP_list) == 0:
            freq_list.append(0)
        else:
            freq_list.append(len(AP_list) / float(lenght1))

        abf.setSweep(i, int(get_value("Command channel")))
        curr_list.append(np.mean(current[stim_start : (stim_start + 100)]))

    frequency_list = freq_list
    current_list = curr_list

    comp_list = np.transpose(np.vstack((np.asarray(current_list), np.asarray(frequency_list))))
    
    try:
        np.savetxt(get_value("Working directory") + "freqs.txt", comp_list)
    except:
        np.savetxt("freqs.txt", comp_list)

    with window("f/I", width=380, height=377):
        #add_button("Plot f/I", callback=plot_fI_callback)
        #add_same_line(spacing=13, name="sameline6")
        #add_same_line(spacing=10, name="sameline7")
        add_plot("Lineplot", height=-1)
        add_line_series("Lineplot", "frequency", list(current_list), list(frequency_list), weight=10, color=[232, 163, 33, 100])
        
        set_window_pos("f/I", 1040, 380)

    #return freq_list, curr_list
'''

def calc_all_spike_params(d,filename,save_path,sshcr,extension):
    full_path = join(save_path,filename + extension)
    if exists(full_path+'.csv'):
        os.remove(full_path+'.csv')

    f=open(full_path+'.csv','a')

    f.write("filename" + "," +
        "strain" + "," +
        "sex" + "," +
        "hemisphere" + "," +
        "cell_type" + "," +
        "region" + "," +
        "X" + "," +
        "Y" + "," +
        "sweep" + "," +
        "pA/pF" + "," +
        "current inj" + "," +
        "APnum" + "," +
        "AP peak" + "," +
        "AP hwdt" + "," +
        "AHP" + ","+
        "threshold" + "," +
        "dV/dt max" + '\n'
        )
    # d is a data object (custom defined class called data)
    dt = 1/d.sampleRate

    try:
        del_com = np.diff(getCommandSweep(d,0))
    except:
        return
    starts = np.where(del_com<0)
    ends = np.where(del_com>0)
    stim_start = starts[0][1]
    stim_end = ends[0][1]

    for sweep in range(d.numSweeps):
        response = getResponseDataSweep(d,sweep) # mV
        command = getCommandSweep(d,sweep) # pA

        # stim_start = stim_start - 1
        stim_length = stim_end - stim_start

        threshat = 0
        half2 = 0
        flag_counter = 0
        ap_counter = 0

        for i in range(int(stim_length)):
            if response[stim_start + i] > -20 and flag_counter > (0.002 / dt):
                st1 = int(stim_start + i)                               #calc peak
                st2 = int(stim_start + i + (0.002 / dt))
                peak = np.max(response[st1 : st2])
                peakat = np.argmax(response[st1 : st2]) + st1           

                dvdt = np.gradient(response, dt * 1000)                 #calculate dvdt_max
                st1 = int(stim_start + i - (0.002 / dt))
                st2 = int(stim_start + i + (0.002 / dt))
                dvdt_max1 = np.max(dvdt[st1 : st2])

                st1 = int(stim_start + i)                               #calculate threshold
                for k in range(int(0.002 / dt)):
                    if dvdt[st1 - k] < 50:
                        thresh = response[st1 - k]
                        threshat = st1 - k
                        break

                st1 = int(stim_start + i) 
                st2 = int(stim_start + i + (0.006 / dt))                #calc AHP
                AHP1 = abs(np.min(response[st1 : st2]) - thresh)
                
                half1 = (peakat - threshat) / 2 + threshat
                st1 = peakat
                st2 = peakat + int(0.002 / dt)
                for i in range(st1,st2):
                    if response[i] < response[int(half1)]:
                        half2 = i
                        break
                hwdt = (half2 - half1) * dt *1000 

                currentinj = command[stim_start+5]
                papf = 2*sweep - 2

                f.write(filename + "," +
                        sshcr[0] + "," +
                        sshcr[1] + "," +
                        sshcr[2] + "," +
                        sshcr[3] + ',' +
                        sshcr[4] + ',' +
                        sshcr[5] + ',' +
                        sshcr[6] + ',' +
                        str(sweep + 1) + "," +
                        str(papf) + "," +
                        str(currentinj) + "," +
                        str(ap_counter) + "," +
                        str(peak) + "," +
                        str(hwdt) + "," +
                        str(AHP1) + "," +
                        str(thresh) + "," +
                        str(dvdt_max1) + '\n'
                        )
                
                ap_counter += 1
                flag_counter = 0
            flag_counter += 1    
    f.close()

'''
def calc_rheobase():

    global rheobase

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = 1/d.sampleRate
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    sweep = 0
    for i in range(abf.sweepCount):
        sweep = i
        abf.setSweep(i, int(get_value("Voltage channel")))
        if np.max(current[:]) > -20:
            break
    
    abf.setSweep(sweep, int(get_value("Command channel")))
    rheobase1 = np.mean(current[stim_start : (stim_start + 100)])

    rheobase = rheobase1

def calc_spike_params():

    global AP_peak
    global AP_hwdt
    global AHP
    global AP_threshold
    global dvdt_max

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = 1/d.sampleRate
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    abf.setSweep(int(get_value("Sweep for spike parameters")), int(get_value("Voltage channel")))
    threshat = 0
    half2 = 0
    for i in range(int(float(lenght1) / dt)):
        if current[stim_start + i] > -20:
            st1 = int(stim_start + i)                               #calc peak
            st2 = int(stim_start + i + (0.002 / dt))
            peak = np.max(current[st1 : st2])
            peakat = np.argmax(current[st1 : st2]) + st1           

            dvdt = np.gradient(current, dt * 1000)                 #calculate dvdt_max
            st1 = int(stim_start + i - (0.002 / dt))
            st2 = int(stim_start + i + (0.002 / dt))
            dvdt_max1 = np.max(dvdt[st1 : st2])

            st1 = int(stim_start + i)                               #calculate threshold
            for k in range(int(0.002 / dt)):
                if dvdt[st1 - k] < 50:
                    thresh = current[st1 - k]
                    threshat = st1 - k
                    break

            st1 = int(stim_start + i) 
            st2 = int(stim_start + i + (0.006 / dt))                #calc AHP
            AHP1 = abs(np.min(current[st1 : st2]) - thresh)
            
            half1 = (peakat - threshat) / 2 + threshat
            st1 = peakat
            st2 = peakat + int(0.002 / dt)
            for i in range(st1,st2):
                if current[i] < current[int(half1)]:
                    half2 = i
                    break
            hwdt = (half2 - half1) * dt *1000

            break
        
    calc_rheobase()

    AP_peak = peak
    AP_hwdt = hwdt
    AHP = AHP1
    AP_threshold = thresh
    dvdt_max = dvdt_max1
    
def write():
    with open(file1, "w") as myfile:
        myfile.write("AP peak: "+ "\t" + "\t" + "\t" + "\t" + str(AP_peak) + "\n")
        myfile.write("AP threshold: "+ "\t" + "\t" + "\t" + "\t" + str(AP_threshold) + "\n")
        myfile.write("AP hwdt: "+ "\t" + "\t" + "\t" + "\t" + str(AP_hwdt) + "\n")
        myfile.write("AHP: "+ "\t" + "\t" + "\t" + "\t" + str(AHP) + "\n")
        myfile.write("dV/dt maximum: "+ "\t" + "\t" + "\t" + "\t" + str(dvdt_max) + "\n")
        myfile.write("membrane tau: "+ "\t" + "\t" + "\t" + "\t" + str(membrane_tau) + "\n")
        myfile.write("membrane capacitance: "+ "\t" + "\t" + "\t" + "\t" + str(membrane_capacitance) + "\n")
        myfile.write("input resistance: "+ "\t" + "\t" + "\t" + "\t" + str(Input_resistance) + "\n")
        myfile.write("resting membrane potential: "+ "\t" + "\t" + "\t" + "\t" + str(resting_membrane_potential) + "\n")
        myfile.write("sag amplitude: "+ "\t" + "\t" + "\t" + "\t" + str(sag_amplitude) + "\n")
        myfile.write("sag timing: "+ "\t" + "\t" + "\t" + "\t" + str(sag_timing) + "\n")
        myfile.write("mean firing frequency: "+ "\t" + "\t" + "\t" + "\t" + str(mean_firing_frequency) + "\n")
        myfile.write("maximum firing frequency: "+ "\t" + "\t" + "\t" + "\t" + str(max_firing_frequency) + "\n")
        myfile.write("rheobase: "+ "\t" + "\t" + "\t" + "\t" + str(rheobase) + "\n")
        myfile.write("accommodation ratio: "+ "\t" + "\t" + "\t" + "\t" + str(accommodation_ratio) + "\n")
        if frequency_list is list:
            myfile.write("firing rate per current density: " + "\n")
            for i in range(len(current_list)):
                myfile.write(str(current_list[i]) + "\t" + str(frequency_list[i]) + "\n")


def write():
    with open(file1, "w") as myfile:
        myfile.write(str(AP_peak) + "\n")
        myfile.write(str(AP_threshold) + "\n")
        myfile.write(str(AP_hwdt) + "\n")
        myfile.write(str(AHP) + "\n")
        myfile.write(str(dvdt_max) + "\n")
        myfile.write(str(membrane_tau) + "\n")
        myfile.write(str(membrane_capacitance) + "\n")
        myfile.write(str(Input_resistance) + "\n")
        myfile.write(str(resting_membrane_potential) + "\n")
        myfile.write(str(sag_amplitude) + "\n")
        myfile.write(str(sag_timing) + "\n")
        myfile.write(str(mean_firing_frequency) + "\n")
        myfile.write(str(max_firing_frequency) + "\n")
        myfile.write(str(rheobase) + "\n")
        myfile.write(str(accommodation_ratio) + "\n")
        if frequency_list is list:
            myfile.write("firing rate per current density: " + "\n")
            for i in range(len(current_list)):
                myfile.write(str(current_list[i]) + "\t" + str(frequency_list[i]) + "\n")
def savetxt():
    try:
        f=open(get_value("Working directory") + 'results.txt','a')
    except:
        f=open('results.txt','a')

    f.write(str(AP_peak) + "\t" +
            str(AP_threshold) + "\t" +
            str(AP_hwdt) + "\t" +
            str(AHP) + "\t" +
            str(dvdt_max) + "\t" +
            str(membrane_tau) + "\t" +
            str(membrane_capacitance) + "\t" +
            str(Input_resistance) + "\t" +
            str(resting_membrane_potential) + "\t" +
            str(sag_amplitude) + "\t" +
            str(sag_timing) + "\t" +
            str(mean_firing_frequency) + "\t" +
            str(max_firing_frequency) + "\t" +
            str(rheobase) + "\t" +
            str(accommodation_ratio) + '\n'
            )
    f.close()

def plot_callback(sender, data):

    for i in range(100):
        try:
            delete_series(plot="Recording", series="sweep" + str(i))
        except:
            pass
  

    xlist = [[] for i in range(abf.sweepCount)]
    ylist = [[] for i in range(abf.sweepCount)]
    for i in range(abf.sweepCount):
        abf.setSweep(i,int(get_value("Voltage channel")))
        xlist[i] = list(abf.sweepX)
        ylist[i] = list(current)

        plot_sweep_name = "sweep" + str(i)

        add_line_series("Recording", plot_sweep_name, xlist[i], ylist[i], weight=2, color=[232, 163, 33, 100])
# def analyzeFile(filename, self):
#     full_file = join(self.directory, filename)
#     abf = pyabf.ABF(full_file)

#     input_cmd = np.empty((abf.sweepCount, 1))
#     AP_count = np.empty((abf.sweepCount, 1))
#     input_dur = np.empty((abf.sweepCount, 1))

#     # Analyze traces
#     for sweep_idx in range(abf.sweepCount):
#         input_cmd[sweep_idx], AP_count[sweep_idx], input_dur[sweep_idx] = countAPs(abf, sweep_idx, self) # x is time, y is data, c is command
#     return input_cmd, AP_count, input_dur

# def countAPs(abf, sweep_idx, self):
#     # Purpose: To count the number of APs in a trace
#     # Inputs: abf struct, which sweep (index), gui self
#     # Outputs: input command (current), number of APs in trace, input duration (for each trace)
#     abf.setSweep(sweep_idx, channel=0)
#     time = abf.sweepX
#     data = current

#     abf.setSweep(sweep_idx, channel=1)
#     command = current

#     # GET COMMAND
#     baseline_cmd = np.array(command[0:10]).mean() #get baseline command (no input)
#     is_on = command > baseline_cmd # Create logical mask for when command input is on
#     input_dur = np.sum(is_on) # number of samples collected with input ON
#     input_cmd = np.array(command[is_on]).mean() # get average of input command

#     # GET NUMBER OF ACTION POTENTIALS
#     baseline_data = np.array(data[0:10]).mean() # get baseline data
#     peaks, _ = sig.find_peaks(data, height=.5*baseline_data, distance=50, rel_height=0)
#     num_APs = len(peaks)

#     return input_cmd, num_APs, input_dur

# def plotMembraneTest(abf):
#     # Purpose: To extract the membrane test parameters from an ABF file
#     # Inputs: N/A
#     # Outputs: N/A
#     memtest = mem.Memtest(abf)
#     fig = plt.figure(figsize=(8, 5))

#     ax1 = fig.add_subplot(221)
#     ax1.grid(alpha=.2)
#     ax1.plot(abf.sweepTimesMin, memtest.Ih.values, 
#             ".", color='C0', alpha=.7, mew=0)
#     ax1.set_title(memtest.Ih.name)
#     ax1.set_ylabel(memtest.Ih.units)

#     ax2 = fig.add_subplot(222)
#     ax2.grid(alpha=.2)
#     ax2.plot(abf.sweepTimesMin, memtest.Rm.values, 
#             ".", color='C3', alpha=.7, mew=0)
#     ax2.set_title(memtest.Rm.name)
#     ax2.set_ylabel(memtest.Rm.units)

#     ax3 = fig.add_subplot(223)
#     ax3.grid(alpha=.2)
#     ax3.plot(abf.sweepTimesMin, memtest.Ra.values, 
#             ".", color='C1', alpha=.7, mew=0)
#     ax3.set_title(memtest.Ra.name)
#     ax3.set_ylabel(memtest.Ra.units)

#     ax4 = fig.add_subplot(224)
#     ax4.grid(alpha=.2)
#     ax4.plot(abf.sweepTimesMin, memtest.CmStep.values, 
#             ".", color='C2', alpha=.7, mew=0)
#     ax4.set_title(memtest.CmStep.name)
#     ax4.set_ylabel(memtest.CmStep.units)

#     for ax in [ax1, ax2, ax3, ax4]:
#         ax.margins(0, .9)
#         ax.set_xlabel("Experiment Time (minutes)")
#         for tagTime in abf.tagTimesMin:
#             ax.axvline(tagTime, color='k', ls='--')
#     plt.tight_layout()
#     plt.show()
#     return
# %%
'''