"""
Regress out signal from non-target rois
"""

curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD

import sys
sys.path.insert(0,curr_dir)

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import pepdoc_params as params
import pdb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pepdoc_params as params
import scipy.io as spio

scaler = MinMaxScaler()

print('libraries loaded...')



#load params for decoding
channels = params.channels
sub_list = params.sub_list
data_dir = params.data_dir
categories = params.categories
labels = params.labels

#timing info
stim_onset = params.stim_onset
bin_size = params.bin_size
bin_length = params.bin_length
start_window = params.start_window
stim_offset = params.stim_offset
suf = '_full'


results_dir = f'{curr_dir}/results' #where to save the results

rois = ['dorsal','ventral','frontal']
control_rois = ['frontal','occipital','partial_brain','dorsal','ventral']

def split_ts(eeg_ts):
    """
    split eeg time series by categories
    """
    #timepoints per category
    start_time = 0
    timepoints = int(eeg_ts.shape[0]/(len(categories)*5))
    end_time = timepoints

    all_cat_data = []
    #split into categories
    for category in categories:
        for nn in range(1,6):
            #extract timepoints for category
            eeg_ts_cat = eeg_ts[start_time:(start_time+timepoints),:]
            start_time = end_time
            end_time = start_time+timepoints

            all_cat_data.append(eeg_ts_cat)
            
    all_cat_data = np.asanyarray(all_cat_data)

    return all_cat_data



for sub in sub_list:
    for roi in rois:
        print(f'{sub} {roi}')
        #load timeseries
        roi_ts = np.load(f'{data_dir}/{sub}/{roi}_concat_ts{suf}.npy')

        #load control rois
        for control in control_rois:
            control_ts = np.load(f'{data_dir}/{sub}/{control}_pcs{suf}.npy')
            


            #loop through channels and regress out signal from control rois
            resid_ts = np.zeros(roi_ts.shape)
            for channel in range(0, roi_ts.shape[1]):
                ols = sm.OLS(roi_ts[:,channel],control_ts).fit() #fit the model

                #extract residuals
                resid_ts[:,channel] = ols.resid
            
            all_cat_data = split_ts(resid_ts)
            
            
            
            #save residuals
            np.save(f'{data_dir}/{sub}/{roi}_{control}_resid_ts{suf}.npy',resid_ts)
            spio.savemat(f'{data_dir}/{sub}/{roi}_{control}_resid_ts{suf}.mat', {f'{sub[1]}_{roi}_ts': resid_ts})
            np.save(f'{data_dir}/{sub}/{roi}_{control}_cat_resid{suf}.npy',all_cat_data)

