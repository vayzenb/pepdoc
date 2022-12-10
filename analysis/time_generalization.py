import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pingouin as pg

import pdb

data_dir = f'/lab_data/behrmannlab/vlad/pepdoc/results_ex1' #read in the file; first value is the file name
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
results_dir = f'{curr_dir}/results' #where to save the results
bin_size = 1 #20 ms bins (EACH BIN IS 4 MS SO 5 ROWS ARE 20 MS)
stim_onset = 0 #stimulus onset value (analysis time is -50, and we use 4 ms bins)
offset_window =138 #when to cut the timecourse
# bin_size = 1 
sub_list = ['AC_newepoch','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ']
categories = ['tool','nontool','bird','insect']
labels = np.asanyarray([0]*5 + [1]*5 + [2]*5 + [3]*5) #creates labels for data

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
            roi_data1 = roi_data1[stim_onset:offset_window,:]
            for roi2 in rois:
                roi_data2 = np.load(f'{results_dir}/rsa/{roi2}_rdm.npy')
                roi_data2 = roi_data2[stim_onset:offset_window,:]

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
                    roi_data1 = roi_data1[stim_onset:offset_window,:]

                    for roi2 in rois:
                        roi_data2 = np.load(f'{data_dir}/{sub}/{roi2}_rdm.npy')
                        roi_data2 = roi_data2[stim_onset:offset_window,:]

                        #calculate time generalization
                        tgm = time_generalization(roi_data1,roi_data2)
                        partial_tgm = partial_time_generalization(roi_data1,roi_data2,control_rdm)
                        
                        np.save(f'{data_dir}/{sub}_{roi1}_{roi2}_corr_ts.npy',tgm)
                        np.save(f'{data_dir}/{sub}_{roi1}_{roi2}_corr_ts_{control_roi}.npy',partial_tgm)

                
time_generalization_sub()