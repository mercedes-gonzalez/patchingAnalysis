from time import time
import pyabf
import matplotlib.pyplot as plt
import time
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.signal as signal
# from scipy.fft import fft, fftfreq
from numpy.fft import fft, ifft
import pandas as pd

onecelllist = pd.read_csv("/Users/mercedesgonzalez/Dropbox (GaTech)/Research/ADfigs/onepercell.csv")
firstcell = onecelllist[onecelllist['first']==1].drop(columns=['first','second']).values.tolist()
firstcell = [item for sublist in firstcell for item in sublist]
secondcell = onecelllist[onecelllist['second']==1].drop(columns=['first','second']).values.tolist()
secondcell = [item for sublist in secondcell for item in sublist]

def makeNewCSV(base_path,csv_name,cellist,numstr):
    df = pd.read_csv(join(base_path,csv_name))
    new_df = df[df['filename'].isin(cellist)]
    
    new_csv_name = csv_name[:-4] + '-'+str(numstr)+'.csv'
    print(new_csv_name)
    new_df.to_csv(join(base_path,new_csv_name),index=False)


base_path = "/Users/mercedesgonzalez/Dropbox (GaTech)/Research/ADfigs/"
makeNewCSV(base_path,"compiled_pas_params.csv",firstcell,1)
makeNewCSV(base_path,"compiled_spike_params.csv",firstcell,1)
makeNewCSV(base_path,"compiled_firing_freq.csv",firstcell,1)

makeNewCSV(base_path,"compiled_pas_params.csv",secondcell,2)
makeNewCSV(base_path,"compiled_spike_params.csv",secondcell,2)
makeNewCSV(base_path,"compiled_firing_freq.csv",secondcell,2)