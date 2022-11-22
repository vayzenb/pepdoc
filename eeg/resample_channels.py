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

iter = 1000 #number of iterations to use for decoding

results_dir = f'{curr_dir}/results' #where to save the results


#read roi from the input
roi = sys.argv[1]



#classifier info
svm_test_size = .4
svm_splits = 20
sss = StratifiedShuffleSplit(n_splits=svm_splits, test_size=svm_test_size)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

iter = 1000
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
 


    #decode_sig = np.asanyarray(decode_sig)
    cat_decode = np.asanyarray(cat_decode)

    return cat_decode

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


def resample_channels(all_subs):
    '''
    Resample channels
    '''
    resampled_data = []
    for sub_data in all_subs:
        #sample channels with replacement
        resampled = np.random.choice(sub_data.shape[2], sub_data.shape[2], replace = True)
        sub_data = sub_data[:,:,resampled]

        resampled_data.append(sub_data)

    return resampled_data

def calc_sig_time(all_sub_decode, test_val):
    '''
    Calculate significant time
    '''
    sig_ts = []
    for time in range(0,all_sub_decode.shape[1]):
        p_val= stats.ttest_1samp(all_sub_decode[:,time], test_val, axis = 0, alternative='greater')
        
        #append the p-value for every time point
        sig_ts.append(p_val[1])  

    

    #reconvert p-value list into a numpy array
    sig_ts = np.asanyarray(sig_ts)
    sig_onset = np.where(sig_ts <=.05,)[0][0]

    return sig_onset*bin_length, sig_ts

'''
load sub data
'''


decode_onset_df = pd.DataFrame(columns = [roi])
erp_onset_df = pd.DataFrame(columns = [roi])


all_subs = []
for sub in sub_list:
    print(sub, roi)
    #load sub data
    #data are formatted stim x time x channels
    curr_sub = np.load(f'{data_dir}/{sub}/{roi}_data.npy')

    #analyze only stim period
    curr_sub = curr_sub[:,stim_onset:,:]

    #convert to pandas dataframe
    
    all_subs.append(curr_sub)

decode_onset_boot = []
erp_onset_boot = []
for ii in range(0,iter):
    
    '''
    resample channels
    '''
    resampled_data = resample_channels(all_subs)
    '''
    decode and extract ERP
    '''
    all_sub_decode = []
    all_sub_erp = []
    for sub_data in enumerate(resampled_data):
        #decode TS and append
        decode_ts = decode_eeg(sub_data[1])
        all_sub_decode.append(decode_ts)

        #calculate mean ERP and append
        #Average roi_data across channels for each object
        roi_erp = extract_erp(sub_data[1])

    #Calc significance of decoding
    all_sub_decode = np.asanyarray(all_sub_decode)
    decode_onset, decode_sig = calc_sig_time(all_sub_decode, test_val=.25)
    decode_onset_boot.append(decode_onset)

    #calc significance of ERP
    all_sub_erp = np.asanyarray(all_sub_erp)
    erp_onset, erp_sig = calc_sig_time(all_sub_erp, test_val=0)
    erp_onset_boot.append(erp_onset)

#add datat to dataframe
decode_onset_df[roi] = decode_onset
erp_onset_df[roi] = erp_onset
'''
save results
'''
decode_onset_df.to_csv(f'{results_dir}/onsets/{roi}_channel_resample_boots.csv')
erp_onset_df.to_csv(f'{results_dir}/onsets/{roi}_erp_boots.csv')
#np.save(f'{results_dir}/{roi}_onset_iter{ii}.npy', onset)


