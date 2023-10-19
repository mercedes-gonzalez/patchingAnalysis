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

save_path = "/Users/mercedesgonzalez/Dropbox (GaTech)/Research/Moscow Rig/FilesFromMoscowRig/patch-walk/connectivity_data/pair-rec/matrix-0512/images/"
abf = pyabf.ABF("/Users/mercedesgonzalez/Dropbox (GaTech)/Research/Moscow Rig/FilesFromMoscowRig/patch-walk/connectivity_data/pair-rec/matrix-0512/pair3/23512033.abf")
myData = abf2class(abf)
length = len(myData.time)
mid = int(round(length/2,1))
dt = myData.time[1]-myData.time[0]

x = getCurrentSweep(myData,0)

if 0:
    x = x[mid:-1] - np.average(x[mid:-1])
    t = myData.time[1:mid]
else:
    xtemp = x - np.average(x)
    ttemp = myData.time

    x = xtemp[int(8.8/dt):int(9.15/dt)]
    t = ttemp[0:int((9.15-8.8)/dt)+1]

    print(len(x))
    print(len(t))


fs = myData.sampleRate  # Sample frequency (Hz)
f0 = 60   # Frequency to be removed from signal (Hz)
Q = 10  # Quality factor

# Design notch filter
b_notch, a_notch = signal.iirnotch(f0, Q, fs)
filt_x = signal.filtfilt(b_notch, a_notch, x)

sr = myData.sampleRate
X = fft(x)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

filt_X = fft(filt_x)

plt.figure(figsize = (12, 6))

plt.subplot(221)
plt.stem(freq, np.abs(X)/len(x)+ np.average(x), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 100)

plt.subplot(222)
plt.plot(t, ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(223)
plt.stem(freq, np.abs(filt_X)/len(filt_x) + np.average(x), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 100)

plt.subplot(224)
plt.plot(t, ifft(filt_X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()

plt.savefig(join(save_path,'23512035.png'), format='png', dpi=300)

plt.show()