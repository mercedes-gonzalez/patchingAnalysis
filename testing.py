from time import time
import pyabf
import matplotlib.pyplot as plt
import time
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import patchAnalysis as pa
import scipy.signal as signal
# from scipy.fft import fft, fftfreq
from numpy.fft import fft, ifft
import scipy.signal as sig

class data:
    def __init__(self,time,current,command,sampleRate,numSweeps):
        self.time = time
        self.current = current
        self.command = command
        self.sampleRate = sampleRate
        self.numSweeps = numSweeps
        
def abf2class(abf):
    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        if sweepNumber == 0:
            myData = data(time=abf.sweepX,current=abf.sweepY,command=abf.sweepC,sampleRate=int(1/(abf.sweepX[1]-abf.sweepX[0])),numSweeps=len(abf.sweepList))
        else:
            myData.current = np.vstack((myData.current,abf.sweepY))
            myData.command = np.vstack((myData.command,abf.sweepC))
    return myData

def getResponseDataSweep(d,sweepNum):
    return d.current[sweepNum,:]

def getCommandSweep(d,sweepNum):
    return d.command[sweepNum,:]

def moving_average(x, w): # for memtest 
    return np.convolve(x, np.ones(w), 'same') / w

base_path = "/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/2023-11-21/"
save_path = "/Users/mercedesgonzalez/Dropbox (GaTech)/Research/ADfigs/currentclamp_pngs/"
# abf_list = [f for f in listdir(base_path) if isfile(join(base_path,f)) & f.endswith(".abf")]
filename = '23n21026.abf'
abf = pyabf.ABF(join(base_path,filename))
myData = abf2class(abf)
print(myData.numSweeps)
del_com = np.diff(getCommandSweep(myData,0))

starts = np.where(del_com<0)
ends = np.where(del_com>0)
stim_start = starts[0][1]
stim_end = ends[0][1]
dt = 1/myData.sampleRate

for sweep in range(myData.numSweeps):
    response = getResponseDataSweep(myData,sweep) # mV
    command = getCommandSweep(myData,sweep) # pA
    
    response_findpeak = response[stim_start:stim_end+500] # make sure to get the edges
    time_findpeak = myData.time[stim_start:stim_end+500]

    peaks, prop = sig.find_peaks(moving_average(response_findpeak,50),prominence=(25,None),height=(.25*(max(response)-min(response))+min(response))) # new method
    prominences, baseL, baseR = sig.peak_prominences(moving_average(response_findpeak,50),peaks=peaks)
    peaks = peaks - 5

    # contour_heights = response_findpeak[peaks] - prominences
    # plt.plot(response_findpeak)
    # plt.plot(peaks, response_findpeak[peaks], "x")
    # plt.vlines(x=peaks, ymin=contour_heights, ymax=response_findpeak[peaks])
    # plt.show()

    window = .003 # this *2 = well outside the limits of the AP
    thresholds=np.empty((len(peaks),2))
    if len(peaks) > 0:
        for p, peak in enumerate(peaks):
            st1 = int(stim_start + peak - (window / dt))
            st2 = int(stim_start + peak + (window / dt))

            # halfwidth
            half_peak_value = response[stim_start + peak]-(0.5*prominences[p])
            thisAPonly = response[st1:st2]
            crossing_indices = np.where(thisAPonly > half_peak_value)[0]

            # Find the time points on both sides of the peak
            half_width_start = crossing_indices[0]
            half_width_end = crossing_indices[-1]
            hwdt = (half_width_end-half_width_start)*dt*1000

            # dvdt max
            dvdt = np.abs(np.gradient(response,dt * 1000))
            dvdt_max = np.max(dvdt[st1 : st2])

            # threshold
            checkthresh = dvdt[stim_start+peak-int(window/dt):stim_start+peak-int(window/10/dt)]
            lt50 = np.where(checkthresh < 50)[0]
            threshold_idx_relative = lt50[-1]
            threshold_mv = response[stim_start+peak-int(window/dt)+threshold_idx_relative]
            threshold_idx = stim_start+peak-int(window/dt)+threshold_idx_relative
            
            # AHP = abs(np.min(response[stim_start+peak:st2])-threshold_mv)
            st1 = int(stim_start + peak) 
            st2 = int(stim_start + peak + (0.006 / dt))                #calc AHP
            AHP = abs(np.min(response[st1 : st2]) - threshold_mv)
            
            # plt.subplots()
            # plt.subplot(2,1,1)
            # plt.plot(checkthresh)
            # plt.subplot(2,1,2)
            # plt.plot(response[stim_start+peak-int(window/dt):stim_start+peak-int(window/10/dt)])
            # plt.scatter(threshold_idx,threshold_mv)
            # plt.show()

            thresholds[p,:] = threshold_idx,threshold_mv

        thresholds = np.array(thresholds)
        plt.plot(response,marker='o')
        # plt.scatter(thresholds[:,0],thresholds[:,1])
        plt.scatter(stim_start+peaks,np.ones_like(peaks)*50,marker="|",color='r')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.tight_layout()
        plt.show()