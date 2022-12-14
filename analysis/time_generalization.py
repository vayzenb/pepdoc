curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
import sys

sys.path.insert(0,curr_dir)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pingouin as pg

import pdb
import pepdoc_params as params

results_dir = params.results_dir #where to save the results
data_dir = params.data_dir
#load params for decoding
channels = params.channels
sub_list = params.sub_list
data_dir = params.data_dir
categories = params.categories
labels = params.labels

#timing info
stim_onset = params.stim_onset
stim_offset = params.stim_offset
bin_size = params.bin_size
bin_length = params.bin_length
rois = ['dorsal','ventral','control', 'left_dorsal', 'right_dorsal', 'left_ventral', 'right_ventral']
rois = ['dorsal','ventral']
control_rois = ['frontal', 'occipital']

def time_generalization(roi_data1,roi_data2):
    #set up empty matrix to hold correlations
    corr_ts = np.zeros((roi_data1.shape[0],roi_data1.shape[0]))


    for time1 in range(0,roi_data1.shape[0]):
        for time2 in range(0,roi_data2.shape[0]):
                time_data1 = roi_data1[time1,:]
                time_data2 = roi_data2[time2,:]
                corr_matrix = np.corrcoef(time_data1,time_data2)
                corr = corr_matrix[0,1]

                corr_ts[time1,time2] = corr
    return corr_ts

def partial_time_generalization(roi_data1,roi_data2,control_rdm):
    #set up empty matrix to hold correlations
    corr_ts = np.zeros((roi_data1.shape[0],roi_data1.shape[0]))
    


    for time1 in range(0,roi_data1.shape[0]):
        for time2 in range(0,roi_data2.shape[0]):
            
                #extract data from each time point
                time_data1 = roi_data1[time1,:]
                time_data2 = roi_data2[time2,:]
                control_data = control_rdm[time1,:]
                
                
                if np.mean(time_data1 == time_data2) == 1:
                    corr = 1
                else:
                    #add all data to a dataframe
                    df = pd.DataFrame({'time1':time_data1,'time2':time_data2,'control':control_data})
                    
                    #run partial correlation
                    corr = pg.partial_corr(data=df,x='time1',y='time2',covar='control',method='pearson')['r'][0]
                    

                #normal correlations


                corr_ts[time1,time2] = corr
                
    return corr_ts

def time_generalization_mean():
    """
    Time generalization for mean RDMs
    """
    print('Time generalization for mean RDMs')
    for control_roi in control_rois:
        control_rdm = np.load(f'{results_dir}/rsa/{control_roi}_rdm.npy')
        for roi1 in rois:
            #load data from each ROI
            roi_data1 = np.load(f'{results_dir}/rsa/{roi1}_rdm.npy')
            roi_data1 = roi_data1[stim_onset:stim_offset,:]
            for roi2 in rois:
                roi_data2 = np.load(f'{results_dir}/rsa/{roi2}_rdm.npy')
                roi_data2 = roi_data2[stim_onset:stim_offset,:]

                #calculate time generalization
                tgm = time_generalization(roi_data1,roi_data2)
                partial_tgm = partial_time_generalization(roi_data1,roi_data2,control_rdm)
                
                
                np.save(f'{results_dir}/rsa/{roi1}_{roi2}_corr_ts.npy',tgm)
                np.save(f'{results_dir}/rsa/{roi1}_{roi2}_corr_ts_partial_{control_roi}.npy',partial_tgm)

def time_generalization_sub():
    """
    Time generalization for individual RDMs
    """
    print('Time generalization for individual RDMs')
    for control_roi in control_rois:
        
        for sub in sub_list:
            print(sub)
            control_rdm = np.load(f'{data_dir}/{sub}/{control_roi}_rdm.npy')
            if (np.isnan(control_rdm).any()):
                continue
            else:
                for roi1 in rois:
                    roi_data1 = np.load(f'{data_dir}/{sub}/{roi1}_rdm.npy')
                    roi_data1 = roi_data1[stim_onset:stim_offset,:]

                    for roi2 in rois:
                        roi_data2 = np.load(f'{data_dir}/{sub}/{roi2}_rdm.npy')
                        roi_data2 = roi_data2[stim_onset:stim_offset,:]

                        #calculate time generalization
                        tgm = time_generalization(roi_data1,roi_data2)
                        partial_tgm = partial_time_generalization(roi_data1,roi_data2,control_rdm)
                        
                        np.save(f'{data_dir}/{sub}_{roi1}_{roi2}_corr_ts.npy',tgm)
                        np.save(f'{data_dir}/{sub}_{roi1}_{roi2}_corr_ts_{control_roi}.npy',partial_tgm)

                
time_generalization_sub()