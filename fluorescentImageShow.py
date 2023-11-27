from platform import java_ver
import numpy as np
import matplotlib.pyplot as plt
from os.path import join,isfile
from os import listdir 
import imageio
import matplotlib.animation as ani
from scipy import signal
import pyabf


if 1: # for gfp
    save_path = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/Conferences/SfN2023'
    strname = 'dic-4x-1'
    dic_img = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/mercimages/'+strname+'/'+strname+'_MMStack_Pos0.ome.tif'
    strname = 'gfp' + strname[3:]
    gfp_img = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/mercimages/'+strname+'/'+strname+'_MMStack_Pos0.ome.tif'

    BW = plt.imread(gfp_img)/255
    BW_adj = (BW - BW.min())/(BW.max()-BW.min())
    print(BW_adj)
    zer = np.zeros(BW.shape)
    RGB = np.stack((zer,BW_adj,zer),2)

    plt.imshow(RGB)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(join(save_path,'EC.png'),bbox_inches='tight')
    plt.draw()
    plt.show()

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

if 0: # for compositing
    fig = plt.figure()

    strname = 'dic-4x-1thurs'
    dic_img = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/mercimages/'+strname+'/'+strname+'_MMStack_Pos0.ome.tif'
    strname = 'gfp' + strname[3:]
    gfp_img = '/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/mercimages/'+strname+'/'+strname+'_MMStack_Pos0.ome.tif'
    
    root_path = "/Users/mercedesgonzalez/Dropbox (GaTech)/Research/ADfigs"
    
    BW1 = plt.imread(dic_img)/255
    BW_adj1 = (BW1 - BW1.min())/(BW1.max()-BW1.min())

    BW2 = plt.imread(gfp_img)/255
    BW_adj2 = (BW2 - BW2.min())/(BW2.max()-BW2.min())-.2
    zer = np.zeros(BW2.shape)
    RGB = np.stack((zer,BW_adj2,zer),2)

    plt.imshow(RGB,alpha=1)
    plt.imshow(BW_adj1,cmap='gray',alpha=.5)
    plt.axis('off')
    plt.tight_layout()

    # plt.savefig(join(root_path,'4x-composite.png'),bbox_inches='tight')
    plt.show()


# '''
#     This script will take an abf file and print all the inputs/output as a loop of matplotlib plots. must screen record to capture. 

# '''
# class data:
#     def __init__(self,time,current,command,sampleRate,numSweeps):
#         self.time = time
#         self.current = current
#         self.command = command
#         self.sampleRate = sampleRate
#         self.numSweeps = numSweeps
        
# def abf2class(abf):
#     for sweepNumber in abf.sweepList:
#         abf.setSweep(sweepNumber)
#         if sweepNumber == 0:
#             myData = data(time=abf.sweepX,current=abf.sweepY,command=abf.sweepC,sampleRate=int(1/(abf.sweepX[1]-abf.sweepX[0])),numSweeps=len(abf.sweepList))
#         else:
#             myData.current = np.vstack((myData.current,abf.sweepY))
#             myData.command = np.vstack((myData.command,abf.sweepC))
#     return myData

# def getCurrentSweep(d,sweepNum):
#     return d.current[sweepNum,:]

# def getCommandSweep(d,sweepNum):
#     return d.command[sweepNum,:]


# abf = pyabf.ABF("/Users/mercedesgonzalez/Dropbox (GaTech)/Research/hAPP AD Project/Data/2023/2023-10-04/23o04007.abf")

# myData = abf2class(abf)

# commandWrite = myData.time.reshape((len(myData.time),1))
# currentWrite = myData.time.reshape((len(myData.time),1))

# fig, ax = plt.subplots(nrows=2, sharex=True,gridspec_kw={'height_ratios': [3, 1]})
# fig.set_size_inches(6,5)

# for i in [4]:
#     ax[0].set_ylim([-100,100])
#     ax[1].set_ylim([-100,2000])
#     ax[1].set_xlabel("Time (s)")
#     ax[0].set_ylabel("Membrane Potential (mV)")
#     ax[1].set_ylabel("Current Injection (pA)")
#     current = getCurrentSweep(myData,i)
#     command = getCommandSweep(myData,i)

#     ax[0].plot(myData.time,current,color='firebrick',linewidth=1)
#     ax[1].plot(myData.time,command,color='royalblue',linewidth=1)

#     # ax[0].cla()
#     # ax[1].cla()
# plt.show()
