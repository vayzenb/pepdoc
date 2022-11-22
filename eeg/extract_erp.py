#%%
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD

import sys
sys.path.insert(0,curr_dir)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import scipy.stats as stats

import pdb
import pepdoc_params as params

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

iter = 1000 #number of iterations to use for decoding


rois = ['occipital','frontal','dorsal','ventral']

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

def extract_erp(sub_data):
    '''
    Extract ERP
    '''
    #average across channels
    roi_mean = np.mean(sub_data, axis = 2)
    #Average roi_data across objects
    roi_mean = np.mean(roi_mean, axis = 0)

    #absolute value
    roi_mean = np.abs(roi_mean)

    pre_stim = roi_mean[:stim_onset] #pull out the pre-stimulus data
    pre_stim = np.mean(pre_stim) #average across prestim timepoints

    roi_norm = roi_mean - pre_stim#subtract the prestimulus average from each timepoint
        
    return roi_norm


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


def calc_erps():
    print('Calculating ERPs...')

    for roi in rois:
        all_sub_data = []
        for sub in sub_list:
            print(f'{sub} {roi}')
            #load data
            sub_data = load_data(sub) #load data for each sub
            
            #select channels for the current roi
            roi_data = select_channels(sub_data, channels[roi])
            #roi_data = np.abs(roi_data) #take the absolute value of the data

            #Average roi_data across channels for each object
            roi_mean = extract_erp(roi_data)

            
            all_sub_data.append(roi_mean)
        
        
        all_sub_data = np.asanyarray(all_sub_data)
        
        #mean_rdm = np.mean(all_sub_rdms, axis=0) #mean across subjects
        
        #save RDM
        np.save(f'{results_dir}/erp/{roi}_mean_ts.npy', all_sub_data)

def calc_cis(iter):
    """
    Calc bootstrap CIs
    """
    print('Calculating CIs...')

    alpha = .05

    boot_df = pd.DataFrame()
    for roi in rois:
        erp_data = np.load(f'{results_dir}/erp/{roi}_mean_ts.npy')

        
        pre_stim = erp_data[:,:stim_onset] #pull out the pre-stimulus data
        pre_stim = np.mean(pre_stim, axis=1) #average across prestim timepoints

        erp_data = erp_data - pre_stim[:,None]#subtract the prestimulus average from each timepoint

        #decoding_data = decoding_data[:,stim_onset:]
        erp_data = pd.DataFrame(erp_data) #convert to dataframe because it has a good resampling function
        erp_boot = []
        
        
        for ii in range(0,iter):
            #resample the sub decode data with replacement
            sub_sample = erp_data.sample(erp_data.shape[0],replace = True, random_state=ii)
            
            #convert it back to a numpy array
            sub_sample = sub_sample.to_numpy() 

            #calculate the bootstrap sample mean
            erp_boot.append(np.mean(sub_sample, axis = 0))

        erp_boot = np.asanyarray(erp_boot)
        erp_cis = np.zeros((3,erp_boot.shape[1]))
        for time in range(0,erp_boot.shape[1]):
            #plot the difference values with confidence intervals
            
            ci_low = np.percentile(erp_boot[:,time], (alpha)*100)
            ci_high= np.percentile(erp_boot[:,time], 100-(alpha)*100)
            if ci_low >= 0:
                sig = 1
            else:
                sig = 0

            erp_cis[0,time] = ci_low
            erp_cis[1,time] = ci_high
            erp_cis[2,time] = sig

        onset_num = np.where(erp_cis[2,:] == 1)[0][0]
        onset = (onset_num *bin_length)-start_window
        print(roi, onset)

        np.save(f'{results_dir}/erp/{roi}_erp_cis.npy',erp_cis)
            
            
def bootstrap_onset(iter):
    '''
    Boot strap participant data and extract onset
    '''
    
    print('Bootstrapping onsets...')

    boot_df = pd.DataFrame()
    for roi in rois:
        roi_erp = np.load(f'{results_dir}/erp/{roi}_mean_ts.npy')
        
        roi_erp = roi_erp[:,stim_onset:]
        erp_data = pd.DataFrame(roi_erp) #convert to dataframe because it has a good resampling function
        erp_boot = []
        sub_counts = np.zeros((1,erp_data.shape[1]))

        for ii in range(0,iter):
            
            #resample the sub decode data with replacement
            sub_sample = erp_data.sample(erp_data.shape[0],replace = True, random_state=ii)
            
            #convert it back to a numpy array
            sub_sample = sub_sample.to_numpy() 
            
            #calculate the bootstrap sample mean
            sig_boot = []
            
            #only add values if they are significant two timepoints in arow
            sig_consistent = []
            
            sig_ts = []
            for time in range(0,sub_sample.shape[1]):
                p_val= stats.ttest_1samp(sub_sample[:,time], 0, axis = 0, alternative='greater')
                
                #append the p-value for every time point
                sig_ts.append(p_val[1])  


                #mark timepoints that are above chance for at least two timepoints in a row
                if time > 0:

                    if sig_ts[time] <= .05 and sig_ts[time-1] <=.05:
                        sig_consistent.append(1)
                    else:
                        sig_consistent.append(0)

            #reconvert p-value list into a numpy array
            sig_ts = np.asanyarray(sig_ts)

            #reconvert consistent list into a numpy array
            sig_consistent = np.asanyarray(sig_consistent)
                
            #find the the first time point that is below change (0.05)
            #np.where simply returns the indices (i.e., spots in an array), that meet some condition
            #i'm simply grabbing the first value of that list, which corresponds to the first time point above chance
            try:
                sig_onset = np.where(sig_consistent ==1,)[0][0]
            except:
                sig_onset= erp_data.shape[1]
            
            
            
            sub_counts[0,np.where(sig_ts <=.05)[0]] += 1
            
            #if d_onset == 1:
            #    pdb.set_trace()
            
            #convert to the actual time point
            sig_onset = (sig_onset *4)

            #add the onset value from the resample to a list
            erp_boot.append(sig_onset)
        
        boot_df[roi] = erp_boot

    boot_df.to_csv(f'{results_dir}/erp/erp_onset_boot.csv',index = False)


calc_erps()
calc_cis(10000)
bootstrap_onset(1000)
