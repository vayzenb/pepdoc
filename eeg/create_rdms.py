curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
import sys

sys.path.insert(0,curr_dir)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pepdoc_params as params
import pdb

data_dir = f'/lab_data/behrmannlab/vlad/pepdoc/results_ex1' #read in the file; first value is the file name
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
results_dir = f'{curr_dir}/results' #where to save the results
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


rois = ['frontal','dorsal','ventral','occipital']

iter = 10000

def load_data(sub):
    

    all_data =[]
    cat_labels = []

    for category in categories: #loop through categories
        for nn in range(1,6): #loop through exemplars in categories
            cat_labels.append(f'{category}{nn}') #append labels for each exemplar
        
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
        
    

    return all_data, cat_labels




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
                
        #pdb.set_trace()
        #rdm = 1-metrics.pairwise.cosine_similarity(curr_time)
        rdm = np.corrcoef(curr_time)
        
        rdm = np.arctanh(rdm)
        rdm_vec = rdm[np.triu_indices(n=curr_time.shape[0],k=1)] #remove lower triangle
        all_rdms.append(rdm_vec*-1)

    return np.asanyarray(all_rdms)

def create_sub_rdms():
    for roi in rois:
        all_sub_rdms = []
        for sub in sub_list:
            print(f'{sub} {roi}')
            #load data
            sub_data,labels = load_data(sub) #load data for each sub
            
            #select channels for the current roi
            roi_data = select_channels(sub_data, channels[roi])
            
            
            
            #create RDM for each timepoint
            rdm = create_rdm(roi_data)

            if (np.isnan(rdm).any()):
                continue
            else:
                np.save(f'{data_dir}/{sub}/{roi}_rdm.npy', rdm)

                #standardize RDMS
                rdm = (rdm - np.mean(rdm, axis=0))/np.std(rdm, axis=0)
                all_sub_rdms.append(rdm)
                
            
        
        
        all_sub_rdms = np.asanyarray(all_sub_rdms)
        
        
        mean_rdm = np.mean(all_sub_rdms, axis=0) #mean across subjects
        #pdb.set_trace()

        #save RDM
        np.save(f'{results_dir}/rsa/{roi}_rdm.npy', mean_rdm)

#

#calculate noise ceiling

def calc_noise_ceiling():
        
    for roi in rois:
        all_sub_rdms = []
        for sub in sub_list:
            
            sub_rdm = np.load(f'{data_dir}/{sub}/{roi}_rdm.npy')

            #standardize RDMS
            sub_rdm = (sub_rdm - np.mean(sub_rdm, axis=0))/np.std(sub_rdm, axis=0)

            all_sub_rdms.append(sub_rdm)


        all_sub_rdms = np.asanyarray(all_sub_rdms)
        boot_corrs = np.zeros((iter, all_sub_rdms.shape[1]))
        for ii in range(0,iter):
            #print(f'iter {ii}')
            #randomly shuffle the RDMs
            np.random.shuffle(all_sub_rdms)
            
            #split into two sets
            set1 = np.mean(all_sub_rdms[0:int(all_sub_rdms.shape[0]/2),:, :], axis=0)
            set2 = np.mean(all_sub_rdms[int(all_sub_rdms.shape[0]/2):,:, :], axis=0)

            for time in range(0, set1.shape[0]):
                boot_corrs[ii,time] = np.corrcoef(set1[time,:], set2[time,:])[0,1]

        np.save(f'{results_dir}/rsa/{roi}_noise_ceiling.npy', boot_corrs)

create_sub_rdms()
#calc_noise_ceiling()