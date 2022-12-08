"""
Creates one concatenated time series for each subject

Outputs in both npy and mat file to support MVGCA

"""


curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
import os
import sys

sys.path.insert(0,curr_dir)
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pdb
import scipy.stats as stats
import pepdoc_params as params

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
stim_offset = params.stim_offset



results_dir = f'{curr_dir}/results' #where to save the results

rois =  ['dorsal','ventral','frontal','occipital']


def concat_data(sub_list):

    all_sub_data = []

    for sub in sub_list: #loop through subs
        all_data =pd.DataFrame()
        

        for category in categories: #loop through categories
            for nn in range(1,6): #loop through exemplars in categories
            
                curr_df = pd.read_csv(f'/{data_dir}/{sub}/{category}s/{category}{nn}.tsv' , sep='\t')#read in the file; first value is the file name
                curr_df = curr_df.T #use pandas to transpose data
                curr_df.columns = curr_df.iloc[0] #set the column names to the first row
                curr_df = curr_df.drop(curr_df.index[0]) #drop the first row
                curr_df = curr_df.astype(float) #convert to float

                bin_data = curr_df.rolling(bin_size).mean() #rolling avg given the bin size
                
                bin_data = bin_data.dropna() #drop missing values

                
                bin_data = bin_data.iloc[stim_onset:stim_offset,:] #cut the pre-stim period, and the post-analysis period
                bin_data = bin_data.reset_index() #reset the index of the dataframe
                
                all_channel_data = bin_data.drop(columns = ['index']) #drop columns that are not channels
                

                #append dummy row to data
                null_data = pd.DataFrame(np.zeros((1, len(all_channel_data.columns))), columns=all_channel_data.columns)
                all_data = pd.concat([all_data, null_data])
                
                #append new object data
                all_data = pd.concat([all_data, all_channel_data])

                
        all_data.reset_index() #reset the index of the dataframe    
        all_sub_data.append(all_data)

    return all_sub_data




def select_channels(sub_data, channels):
    '''
    Select channels
    '''
    
    channels  =[f'E{ii}' for ii in channels] #convert channels into the same format as the columns

    
    channel_df = pd.DataFrame()
    
    for ii in channels: #loop through all channels of interest
        
        if ii in sub_data.columns: #check if current channel exists in df
            channel_df[ii] = sub_data[ii] #if it does add it the empty one
    
    #channel_df = channel_df.T #tranpose so data are channels x time
    #channel_data.append(df)

    return np.asanyarray(channel_df)



all_sub_data = concat_data(sub_list)

for roi in rois:
    roi_sub_data = []
    for sub in enumerate(sub_list):
        print(f'Extracting for {roi} {sub}')
        roi_data = select_channels(all_sub_data[sub[0]], channels[roi])
        np.save(f'{data_dir}/{sub[1]}_{roi}_concat_ts.npy', roi_data)
        spio.savemat(f'{data_dir}/{sub[1]}_{roi}_concat_data.mat', {f'{sub[1]}_{roi}_ts': roi_data})
        