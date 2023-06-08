from platform import java_ver
import numpy as np
import matplotlib.pyplot as plt
from os.path import join,isfile
from os import listdir 
import imageio
import matplotlib.animation as ani
from scipy import signal
import pyabf


if 0: # for gfp
    root_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2022-06-30/images/'
    img_name = 'patched-gfp5.tif'

    BW = plt.imread(join(root_path,img_name))/255
    BW_adj = (BW - BW.min())/(BW.max()-BW.min())-.15
    print(BW_adj)
    zer = np.zeros(BW.shape)
    RGB = np.stack((zer,BW_adj,zer),2)

    plt.imshow(RGB)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_name+'enh.png',bbox_inches='tight')
    plt.draw()

if 0: # make gif
    ims = []
    fig = plt.figure()

    root_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2022-06-30/images/stack/'
    img_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) and f.endswith(".tif")]
    print(img_list)
    for i in img_list:
        BW = plt.imread(join(root_path,i))/255
        BW_adj = (BW - BW.min())/(BW.max()-BW.min())-.10
        zer = np.zeros(BW.shape)
        RGB = np.stack((zer,BW_adj,zer),2)
        
        plt.axis('off')
        plt.tight_layout()
        im = plt.imshow(RGB,animated=True)
        ims.append([im])

    myAnim = ani.ArtistAnimation(fig, ims, interval=1000,blit=True,repeat_delay=1000)

    # myAnim.save('zstack.mp4')
    plt.show()

if 0: # for gfp
    n = 3
    gfp_img = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/merc images/2023-06-05/gfp'+str(n)+'/Pos0/img_000000000_Default_000.tif'
    root_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/merc images/2023-06-05/'
    BW2 = plt.imread(gfp_img)/255
    BW_adj2 = (BW2 - BW2.min())/(BW2.max()-BW2.min())-.2

    plt.imshow(BW_adj2,alpha=1,cmap='gray')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(join(root_path,str(n)+'gfp.png'),bbox_inches='tight')
    plt.show()

if 0: # for compositing
    dic_img = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/merc images/2023-06-06/dic 4x/Pos0/img_000000000_Default_000.tif'
    gfp_img = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/merc images/2023-06-06/gfp 4x/Pos0/img_000000000_Default_000.tif'
    # dic_img = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/merc images/2023-06-05/dic'+str(n)+'/Pos0/img_000000000_Default_000.tif'
    # gfp_img = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/merc images/2023-06-05/gfp'+str(n)+'/Pos0/img_000000000_Default_000.tif'
    root_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/merc images/2023-06-05/'
    BW1 = plt.imread(dic_img)/255
    BW_adj1 = (BW1 - BW1.min())/(BW1.max()-BW1.min())

    BW2 = plt.imread(gfp_img)/255
    BW_adj2 = (BW2 - BW2.min())/(BW2.max()-BW2.min())-.2
    zer = np.zeros(BW2.shape)
    RGB = np.stack((zer,BW_adj2,zer),2)

    plt.imshow(RGB,alpha=1)
    plt.imshow(BW_adj1,cmap='gray',alpha=.3)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(join(root_path,'4x-composite.png'),bbox_inches='tight')
    plt.show()

# ############## MINI ANIMATION
# from time import time
# import pyabf
# import matplotlib.pyplot as plt
# import time
# import imageio
# from os import listdir
# from os.path import isfile, join
# import numpy as np
# import patchAnalysis as pa
# import scipy.signal as sig

# '''
#     This script will take an abf file and print all the inputs/output as a loop of matplotlib plots. must screen record to capture. 

# '''
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


abf = pyabf.ABF("/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/2023-06-05/23605011.abf")

myData = abf2class(abf)

commandWrite = myData.time.reshape((len(myData.time),1))
currentWrite = myData.time.reshape((len(myData.time),1))

# remove 60 hz lololol
# fs = myData.sampleRate  # Sample frequency (Hz)
# f0 = 60    # Frequency to be removed from signal (Hz)
# Q = 3  # Quality factor
# # Design notch filter
# b_notch, a_notch = signal.iirnotch(f0, Q, fs)


fig, ax = plt.subplots(nrows=2, sharex=True,gridspec_kw={'height_ratios': [3, 1]})
fig.set_size_inches(6,5)

print(myData.numSweeps)
for i in range(myData.numSweeps):
    ax[0].set_ylim([-100,100])
    ax[1].set_ylim([-100,2000])
    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("Membrane Potential (mV)")
    ax[1].set_ylabel("Current Injection (pA)")
    current = getCurrentSweep(myData,i)
    command = getCommandSweep(myData,i)

    # filteredcurrent = signal.filtfilt(b_notch, a_notch, current)
    ax[0].plot(myData.time,current,color='firebrick',linewidth=1)
    ax[1].plot(myData.time,command,color='royalblue',linewidth=1)

    plt.pause(.25)

    ax[0].cla()
    ax[1].cla()
plt.show()


# from numpy.fft import fft, ifft

# abf = pyabf.ABF("/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/2023-06-05/23605011.abf")
# myData = abf2class(abf)
# x = getCurrentSweep(myData,1)
# t = myData.time

# sr = myData.sampleRate
# X = fft(x)
# N = len(X)
# n = np.arange(N)
# T = N/sr
# freq = n/T 

# plt.figure(figsize = (12, 6))
# plt.subplot(121)

# plt.stem(freq, np.abs(X), 'b', \
#          markerfmt=" ", basefmt="-b")
# plt.xlabel('Freq (Hz)')
# plt.ylabel('FFT Amplitude |X(freq)|')
# plt.xlim(0, 1000)

# plt.subplot(122)
# plt.plot(t, ifft(X), 'r')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()
