curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc'
import sys
sys.path.insert(0,curr_dir)
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
scaler = MinMaxScaler()
import pepdoc_params as params

print('libraries loaded...')

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
start_window = params.start_window
stim_offset = params.stim_offset
suf = params.suf
result_suf = '_full'

rois =  ['dorsal','ventral','frontal','occipital']
target_rois = ['dorsal','ventral']

print(sub_list)


def concat_data(sub_list):

    """
    Load subject data and concatenate into one dataframe
    """
    print('Concatenating data...')
    all_sub_data = []

    for sub in sub_list: #loop through subs
        all_data =pd.DataFrame()
        

        for category in categories: #loop through categories
            for nn in range(1,6): #loop through exemplars in categories
            
                curr_df = pd.read_csv(f'/{data_dir}/{sub}/{category}s/{category}{nn}{suf}.csv')#read in the file; first value is the file name
                curr_df = curr_df.T #use pandas to transpose data
                curr_df.columns = curr_df.iloc[0] #set the column names to the first row
                curr_df = curr_df.drop(curr_df.index[0]) #drop the first row
                curr_df = curr_df.astype(float) #convert to float

                bin_data = curr_df.rolling(bin_size).mean() #rolling avg given the bin size
                
                bin_data = bin_data.dropna() #drop missing values

                
                #bin_data = bin_data.iloc[stim_onset:stim_offset,:] #cut the pre-stim period, and the post-analysis period
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
    

    return channel_df

def remove_channels(sub_data, channels):
    '''
    Remove channels
    '''
    
    channels  =[f'E{ii}' for ii in channels] #convert channels into the same format as the columns
    
    for ii in channels: #loop through all channels of interest
        
        if ii in sub_data.columns: #check if current channel exists in df
            sub_data = sub_data.drop(columns = [ii]) #if it does remove it from the df
    

    return sub_data

def run_pca(sub_data,var_perc):
    '''
    Run PCA on data
    '''
    pca = PCA(n_components = var_perc) # variance explained
    norm_data = scaler.fit_transform(sub_data) #scale the data between 0 and 1
    norm_data = pd.DataFrame(norm_data, columns = sub_data.columns) #convert to dataframe
    pca.fit(norm_data) #fit the pca

    X = pca.fit_transform(norm_data) 
    pc_cols = [f'PC{ii}' for ii in range(1, len(X[0])+1)] # create column names for PCs

    #create dataframe with PC loadings
    loadings = pd.DataFrame(pca.components_.T, columns=pc_cols, index=sub_data.columns)
    loadings = loadings.T


    return X, loadings

def create_loadings_rdm(loadings):
    '''
    Create RDM from loadings
    '''
    #create RDM of PC loading similarity
    loading_mat = np.zeros((256,256))
    
    for ii in range(1,257):
        for jj in range(1,257):
            try:
                loading_mat[ii-1,jj-1] = euclidean_distances(loadings[f'E{ii}'].values.reshape(1, -1), loadings[f'E{jj}'].values.reshape(1, -1))[0][0]
            except:
                loading_mat[ii-1,jj-1] = np.nan

    #standarize distance matrix
    loading_mat = (loading_mat - np.nanmean(loading_mat))/np.nanstd(loading_mat)
    return loading_mat

#create concatenated data with all subs
all_sub_data = concat_data(sub_list)
all_sub_mat = np.zeros((256,256))
count_mat = np.zeros((256,256))+1
print(sub_list)

ss = 0
for sub_data in all_sub_data:
    print(sub_list[ss])
    #pdb.set_trace()
    channel_data = pd.DataFrame()
    try:
        sub_data = sub_data.drop(columns = ['E257']) #drop columns that are not channels
    except:
        pass

 
    #extract loadings for whole brain
    eeg_pc, loadings = run_pca(sub_data,.95)
    
    loading_mat = create_loadings_rdm(loadings)
    
    #add loadings from each participant to a matrix
    all_sub_mat = np.nansum(np.dstack((all_sub_mat,loading_mat)),2)
    count_mat[np.where(np.isnan(loading_mat) == False)] = count_mat[np.where(np.isnan(loading_mat) == False)] + 1    


    #run pca on ROIs only
    for roi in rois:
        curr_channel_data = select_channels(sub_data, channels[roi])
        channel_data = pd.concat([channel_data, curr_channel_data], axis = 1)

        eeg_pc, _ = run_pca(channel_data,.95)

        #save pcs
        np.save(f'{data_dir}/{sub_list[ss]}/{roi}_pcs{result_suf}.npy', eeg_pc)
    
    
    

    #run pca with target ROIs REMOVED
    # PCA of the rest of hte brain
    reduced_data = sub_data
    for roi in target_rois:
        reduced_data = remove_channels(reduced_data, channels[roi])

        #run pca on reduced data
        reduced_pc, _ = run_pca(reduced_data,.95)

    #save reduced data
    np.save(f'{data_dir}/{sub_list[ss]}/partial_brain_pcs{result_suf}.npy', reduced_pc)
    
    ss = ss+ 1
    

    

#create mean loading matrix
mean_mat = all_sub_mat/(count_mat)
np.save(f'{results_dir}/channel_similarity{result_suf}.npy', mean_mat)
    

            
            