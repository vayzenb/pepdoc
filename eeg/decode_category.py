curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD

import sys
sys.path.insert(0,curr_dir)

import os
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




#classifier info
svm_test_size = .4
svm_splits = 100
sss = StratifiedShuffleSplit(n_splits=svm_splits, test_size=svm_test_size)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#clf = make_pipeline(StandardScaler(), GaussianNB())

def load_data(sub_list):
    
    all_sub_data = []

    for sub in sub_list: #loop through subs
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
            
        all_sub_data.append(all_data)

    return all_sub_data

def decode_eeg(sub_data):

    '''
    Decode from channels
    '''
    
    cat_decode = []
    decode_sig = [] 
    for time in range(0, sub_data.shape[1]):
        X = sub_data[:,time,:] #grab all data for that time point
        y = labels #set Y to be the labels
        
        temp_acc = [] #create empty list accuracy for each timepoint
        for train_index, test_index in sss.split(X, y): #grab indices for training and test

            X_train, X_test = X[train_index], X[test_index] 
            y_train, y_test = y[train_index], y[test_index]

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            clf.fit(X_train, y_train)   

            temp_acc.append(clf.score(X_test, y_test))

        cat_decode.append(np.mean(temp_acc))
        t_stat = stats.ttest_1samp(temp_acc, .25, axis = 0, alternative='greater')
        decode_sig.append(t_stat[1])

    decode_sig = np.asanyarray(decode_sig)
    #decode_sig = decode_sig[10:]
    onset = np.where(decode_sig <= .05)[0][0]


    #decode_sig = np.asanyarray(decode_sig)
    cat_decode = np.asanyarray(cat_decode)

    return cat_decode, decode_sig

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



def run_decoding():
    print('Decoding categories...')
    all_sub_data = load_data(sub_list)


    for roi in rois:
        roi_decoding = []
        roi_sig = []
        for sub in range(0, len(all_sub_data)):
            print('Decoding: ', sub, roi)
            roi_data = select_channels(all_sub_data[sub], channels[roi])
            np.save(f'/{data_dir}/{sub_list[sub]}/{roi}_data.npy', roi_data)
            decode_results, decode_sig = decode_eeg(roi_data)
            roi_decoding.append(decode_results)
            roi_sig.append(decode_sig)
        
        roi_decoding = np.asanyarray(roi_decoding)
        roi_sig = np.asanyarray(roi_sig)
        np.save(f'{results_dir}/decode/{roi}_decoding.npy', roi_decoding)
        np.save(f'{results_dir}/decode/{roi}_decoding_sig.npy', roi_sig)

def calc_cis(iter):
    """
    Calc bootstrap CIs
    """
    print('Calculating CIs...')

    alpha = .05

    boot_df = pd.DataFrame()
    for roi in rois:
        decoding_data = np.load(f'{results_dir}/decode/{roi}_decoding.npy')
        #decoding_data = decoding_data[:,stim_onset:]
        decoding_data = pd.DataFrame(decoding_data) #convert to dataframe because it has a good resampling function
        decode_boot = []
        

        
        for ii in range(0,iter):
            #resample the sub decode data with replacement
            sub_sample = decoding_data.sample(decoding_data.shape[0],replace = True, random_state=ii)
            
            #convert it back to a numpy array
            sub_sample = sub_sample.to_numpy() 

            #calculate the bootstrap sample mean
            decode_boot.append(np.mean(sub_sample, axis = 0))

        decode_boot = np.asanyarray(decode_boot)
        decode_cis = np.zeros((3,decode_boot.shape[1]))
        for time in range(0,decode_boot.shape[1]):
            #plot the difference values with confidence intervals
            
            ci_low = np.percentile(decode_boot[:,time], (alpha)*100)
            ci_high= np.percentile(decode_boot[:,time], 100-(alpha)*100)
            if ci_low >= .25:
                sig = 1
            else:
                sig = 0

            decode_cis[0,time] = ci_low
            decode_cis[1,time] = ci_high
            decode_cis[2,time] = sig

        onset_num = np.where(decode_cis[2,:] == 1)[0][0]
        onset = (onset_num *bin_length)-start_window
        print(roi, onset)

        np.save(f'{results_dir}/decode/{roi}_decode_cis.npy',decode_cis)

def bootstrap_onset(iter):
    '''
    Boot strap participant data and extract onset
    '''
    print('Bootstrapping onsets...')

    boot_df = pd.DataFrame()
    for roi in rois:
        decoding_data = np.load(f'{results_dir}/decode/{roi}_decoding.npy')
        decoding_data = decoding_data[:,stim_onset:]
        decoding_data = pd.DataFrame(decoding_data) #convert to dataframe because it has a good resampling function
        decode_boot = []
        sub_counts = np.zeros((1,decoding_data.shape[1]))

        for ii in range(0,iter):
            
            #resample the sub decode data with replacement
            sub_sample = decoding_data.sample(decoding_data.shape[0],replace = True, random_state=ii)
            
            #convert it back to a numpy array
            sub_sample = sub_sample.to_numpy() 
            
            #calculate the bootstrap sample mean
            sig_boot = []
            
            #only add values if they are significant two timepoints in arow
            sig_consistent = []
            
            sig_ts = []
            for time in range(0,sub_sample.shape[1]):
                p_val= stats.ttest_1samp(sub_sample[:,time], .25, axis = 0, alternative='greater')
                
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
                sig_onset= decoding_data.shape[1]
            
            
            
            sub_counts[0,np.where(sig_ts <=.05)[0]] += 1
            
            #if d_onset == 1:
            #    pdb.set_trace()
            
            #convert to the actual time point
            sig_onset = (sig_onset *4)

            #add the onset value from the resample to a list
            decode_boot.append(sig_onset)
        
        boot_df[roi] = decode_boot

    boot_df.to_csv(f'{results_dir}/onsets/decode_onset_boot.csv',index= False)


run_decoding()
calc_cis(10000)
bootstrap_onset(10000)