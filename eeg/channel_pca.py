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




data_dir = f'/lab_data/behrmannlab/vlad/pepdoc/results_ex1' #read in the file; first value is the file name
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
results_dir = f'{curr_dir}/results' #where to save the results
bin_size = 1 #20 ms bins (EACH BIN IS 4 MS SO 5 ROWS ARE 20 MS)
stim_onset = 13 #stimulus onset value (analysis time is -50, and we use 4 ms bins)
offset_window =87 #when to cut the timecourse
# bin_size = 1 
categories = ['tool','nontool','bird','insect']
labels = np.asanyarray([0]*5 + [1]*5 + [2]*5 + [3]*5) #creates labels for data

#sub codes
sub_list = ['AC_newepoch','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ']

rois = ['dorsal','ventral','control', 'left_dorsal', 'right_dorsal', 'left_ventral', 'right_ventral']
rois = ['dorsal','ventral','frontal','occipital']

#channels
#channels
channels = {'left_dorsal': [77, 78, 79, 80, 86, 87, 88, 89, 98, 99, 100, 110, 109, 118],
            'right_dorsal': [131, 143, 154, 163, 130, 142, 153, 162, 129, 141, 152, 128, 140, 127],
            'dorsal':  [77, 78, 79, 80, 86, 87, 88, 89, 98, 99, 100, 109, 110, 118] + [131, 143, 154, 163, 130, 142, 153, 162, 129, 141, 152, 128, 140, 127],
            'left_ventral':[104, 105, 106, 111, 112, 113, 114, 115, 120, 121, 122, 123, 133, 134],
            'right_ventral':[169, 177, 189, 159, 168, 176, 18, 199, 158, 167, 175, 187, 166, 174],
            'ventral': [104, 105, 106, 111, 112, 113, 114, 115, 120, 121, 122, 123, 133, 134] + [169, 177, 189, 159, 168, 176, 188, 199, 158, 167, 175, 187, 166, 174],
            'frontal': [11, 12, 18, 19, 20, 21, 25, 26, 27, 32, 33, 34, 37, 38],
            'occipital': [145,146,17,135,136,137,124,125,138,149,157,156,165]}


pca = PCA(n_components = .95) #95% variance explained


def concat_data(sub_list):

    """
    Load subject data and concatenate into one dataframe
    """

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

                
                bin_data = bin_data.iloc[stim_onset:offset_window,:] #cut the pre-stim period, and the post-analysis period
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

    return channel_df

#create concatenated data with all subs
all_sub_data = concat_data(sub_list)
all_sub_mat = np.zeros((256,256))
count_mat = np.zeros((256,256))+1
for sub_data in all_sub_data:
    #pdb.set_trace()
    channel_data = pd.DataFrame()
    try:
        sub_data = sub_data.drop(columns = ['E257']) #drop columns that are not channels
    except:
        continue

    for roi in rois:
        curr_channel_data = select_channels(sub_data, channels[roi])
        channel_data = pd.concat([channel_data, curr_channel_data], axis = 1)
    
    norm_data = scaler.fit_transform(sub_data) #scale the data between 0 and 1
    norm_data = pd.DataFrame(norm_data, columns = sub_data.columns) #convert to dataframe
    pca.fit(norm_data) #fit the pca

    X = pca.fit_transform(norm_data)
    pc_cols = [f'PC{ii}' for ii in range(1, len(X[0])+1)] # create column names for PCs

    #create dataframe with PC loadings
    loadings = pd.DataFrame(pca.components_.T, columns=pc_cols, index=sub_data.columns)
    loadings = loadings.T
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
    
    all_sub_mat = np.nansum(np.dstack((all_sub_mat,loading_mat)),2)
    count_mat[np.where(np.isnan(loading_mat) == False)] = count_mat[np.where(np.isnan(loading_mat) == False)] + 1

mean_mat = all_sub_mat/(count_mat)
np.save(f'{results_dir}/channel_similarity.npy', mean_mat)
    

            
            


'''
running pca on only dorsal and ventral channels 
    #pdb.set_trace()
    channel_data = pd.DataFrame()

    for roi in rois:
        curr_channel_data = select_channels(sub_data, channels[roi])
        channel_data = pd.concat([channel_data, curr_channel_data], axis = 1)
    
    pca.fit(channel_data)

    X = pca.fit_transform(channel_data)

    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=channel_data.columns)

    break
'''

