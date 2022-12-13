"""
Runs decoding on dorsal/ventral channels after other regions are regressed out

"""


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

suf = params.suf

#classifier info
svm_test_size = .4
svm_splits = 100
sss = StratifiedShuffleSplit(n_splits=svm_splits, test_size=svm_test_size)

rois = ['dorsal','ventral']
control_rois = ['partial_brain','frontal','occipital','dorsal','ventral']


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
    #onset = np.where(decode_sig <= .05)[0][0]


    #decode_sig = np.asanyarray(decode_sig)
    cat_decode = np.asanyarray(cat_decode)

    return cat_decode, decode_sig

def run_decoding():

    for roi in rois:
        for control in control_rois:
            control_decode = []
            for sub in sub_list:
                print(f'{sub} {roi} {control} decoding...')

                #load category data
                sub_data = np.load(f'{data_dir}/{sub}/{roi}_{control}_cat_resid.npy')
                sub_data = sub_data[:,0:-1,:] #remove dummy timepoint at the end
                
                

                #run decoding
                cat_decode, decode_sig = decode_eeg(sub_data)

                control_decode.append(cat_decode)

            control_decode = np.asanyarray(control_decode)
            np.save(f'{results_dir}/decode/{roi}_{control}_decode.npy', control_decode)



def calc_cis(iter):
    """
    Calc bootstrap CIs
    """
    print('Calculating CIs...')

    alpha = .05

    boot_df = pd.DataFrame()
    for roi in rois:
        for control in control_rois:
            decoding_data = np.load(f'{results_dir}/decode/{roi}_{control}_decode.npy')
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

            np.save(f'{results_dir}/decode/{roi}_{control}_decode_cis.npy',decode_cis)

def bootstrap_onset(iter):
    '''
    Boot strap participant data and extract onset
    '''
    print('Bootstrapping onsets...')

    
    for roi in rois:
        for control in control_rois:
            boot_df = pd.DataFrame()
            decoding_data = np.load(f'{results_dir}/decode/{roi}_{control}_decode.npy')
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
            
            boot_df[f'{roi}_{control}'] = decode_boot

            boot_df.to_csv(f'{results_dir}/onsets/{roi}_{control}_decode_onset_boot.csv',index= False)


run_decoding()
calc_cis(10000)
bootstrap_onset(10000)
