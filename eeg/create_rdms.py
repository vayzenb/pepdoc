import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

import pdb

data_dir = f'/lab_data/behrmannlab/vlad/pepdoc/results_ex1' #read in the file; first value is the file name
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
results_dir = f'{curr_dir}/results' #where to save the results
bin_size = 1 #20 ms bins (EACH BIN IS 4 MS SO 5 ROWS ARE 20 MS)
stim_onset = 0 #stimulus onset value (analysis time is -50, and we use 4 ms bins)
offset_window =138 #when to cut the timecourse
# bin_size = 1 
categories = ['tool','nontool','bird','insect']
labels = np.asanyarray([0]*5 + [1]*5 + [2]*5 + [3]*5) #creates labels for data

#sub codes
sub_list = ['AC_newepoch','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ']

rois = ['dorsal','ventral','control', 'left_dorsal', 'right_dorsal', 'left_ventral', 'right_ventral']
rois = ['dorsal','ventral']

#channels
channels = {'left_dorsal': [77, 78, 79, 80, 86, 87, 88, 89, 98, 99, 100, 110, 109, 118],
            'right_dorsal': [131, 143, 154, 163, 130, 142, 153, 162, 129, 141, 152, 128, 140, 127],
            'dorsal':  [77, 78, 79, 80, 86, 87, 88, 89, 98, 99, 100, 110, 109, 118] + [131, 143, 154, 163, 130, 142, 153, 162, 129, 141, 152, 128, 140, 127],
            'left_ventral':[104, 105, 106, 111, 112, 113, 114, 115, 120, 121, 122, 123, 133, 134],
            'right_ventral':[169, 177, 189, 159, 168, 176, 18, 199, 158, 167, 175, 187, 166, 174],
            'ventral': [104, 105, 106, 111, 112, 113, 114, 115, 120, 121, 122, 123, 133, 134] + [169, 177, 189, 159, 168, 176, 18, 199, 158, 167, 175, 187, 166, 174],
            'control': [11, 12, 18, 19, 20, 21, 25, 26, 27, 32, 33, 34, 37, 38]}


def load_data(sub):
    

    all_data =[]

    for category in categories: #loop through categories
        for nn in range(1,6): #loop through exemplars in categories
        
            curr_df = pd.read_csv(f'/{data_dir}/{sub}/{category}s/{category}{nn}.tsv' , sep='\t')#read in the file; first value is the file name
            curr_df = curr_df.T #use pandas to transpose data
            curr_df.columns = curr_df.iloc[0] #set the column names to the first row
            curr_df = curr_df.drop(curr_df.index[0]) #drop the first row
            curr_df = curr_df.astype(float) #convert to float

            bin_data = curr_df.rolling(bin_size).mean() #rolling avg given the bin size
            
            bin_data = bin_data.dropna() #drop missing values
            bin_data = bin_data.reset_index() #reset the index of the dataframe
            
            all_channel_data = bin_data.drop(columns = ['index']) #drop columns that are not channels
            all_data.append(all_channel_data)
        
    

    return all_data




def select_channels(sub_data, channels):
    '''
    Select channels
    '''
    channels  =[f'E{ii}' for ii in channels] #convert channels into the same format as the columns

    channel_data = []
    for im_n in range(0, len(sub_data)):
        df = pd.DataFrame()
        for ii in channels: #loop through all channels of interest
            if ii in sub_data[im_n].columns: #check if current channel exists in df
                df[ii] = sub_data[im_n][ii] #if it does add it the empty one
        
        channel_data.append(df)

    return np.asanyarray(channel_data)

def create_rdm(data):
    '''
    Create RDM
    '''
    
    all_rdms = []
    for time in range(data.shape[1]):
        curr_time = data[:,time,:]
        rdm = 1-metrics.pairwise.cosine_similarity(curr_time)
        rdm_vec = rdm[np.triu_indices(n=curr_time.shape[0],k=1)] #remove lower triangle
        all_rdms.append(rdm_vec)

    return np.asanyarray(all_rdms)

for roi in rois:
    all_sub_rdms = []
    for sub in sub_list:
        print(f'{sub} {roi}')
        #load data
        sub_data = load_data(sub) #load data for each sub
        
        #select channels for the current roi
        roi_data = select_channels(sub_data, channels[roi])
        
        
        #create RDM for each timepoint
        rdm = create_rdm(roi_data)
        np.save(f'{data_dir}/{sub}/{roi}_rdm.npy', rdm)

        #standardize RDMS
        #rdm = (rdm - np.mean(rdm, axis=0))/np.std(rdm, axis=0)
        all_sub_rdms.append(rdm)
    
    
    all_sub_rdms = np.asanyarray(all_sub_rdms)
    mean_rdm = np.mean(all_sub_rdms, axis=0) #mean across subjects

    #save RDM
    np.save(f'{results_dir}/rsa/{roi}_rdm.npy', mean_rdm)
        
    