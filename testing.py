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

def getCurrentSweep(d,sweepNum):
    return d.current[sweepNum,:]

def getCommandSweep(d,sweepNum):
    return d.command[sweepNum,:]

base_path = "/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/2023-07-25/"
save_path = "/Users/mercedesgonzalez/Dropbox (GaTech)/Research/ADfigs/currentclamp_pngs/"
abf_list = [f for f in listdir(base_path) if isfile(join(base_path,f)) & f.endswith(".abf")]

for a in abf_list:
    print("File: ",a)
    abf = pyabf.ABF(join(base_path,a))
    myData = abf2class(abf)
    print("Numsweeps: ",myData.numSweeps)

    if myData.numSweeps <= 5:
        print("\tnot enough sweeps.")
    else:
        length = len(myData.time)
        mid = int(round(length/2,1))
        
        plt.figure(figsize = (12, 6))

        for i in range(myData.numSweeps):
            if np.mod(i,3) == 0:
                x = getCurrentSweep(myData,i)
                t = myData.time
                plt.plot(t, x,linewidth=.5)

        plt.xlabel('Time (s)')
        plt.ylabel('membrane potential (mV)')
        plt.title(a)
        plt.tight_layout()

        plt.savefig(join(save_path,a+'.png'))
        plt.clf()