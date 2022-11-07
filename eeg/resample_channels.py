import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pdb
import scipy.stats as stats

print('libraries loaded...')

data_dir = f'/lab_data/behrmannlab/vlad/pepdoc/results_ex1' #read in the file; first value is the file name
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
results_dir = f'{curr_dir}/results' #where to save the results

# bin_size = 1 
categories = ['tool','nontool','bird','insect']
labels = np.asanyarray([0]*5 + [1]*5 + [2]*5 + [3]*5) #creates labels for data

#sub codes
sub_list = ['AC_newepoch','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ']

rois = ['dorsal','ventral','control', 'left_dorsal', 'right_dorsal', 'left_ventral', 'right_ventral']
rois = ['dorsal','ventral', 'occipital','frontal']


#channels
channels = {'left_dorsal': [77, 78, 79, 80, 86, 87, 88, 89, 98, 99, 100, 110, 109, 118],
            'right_dorsal': [131, 143, 154, 163, 130, 142, 153, 162, 129, 141, 152, 128, 140, 127],
            'dorsal':  [77, 78, 79, 80, 86, 87, 88, 89, 98, 99, 100, 109, 110, 118] + [131, 143, 154, 163, 130, 142, 153, 162, 129, 141, 152, 128, 140, 127],
            'left_ventral':[104, 105, 106, 111, 112, 113, 114, 115, 120, 121, 122, 123, 133, 134],
            'right_ventral':[169, 177, 189, 159, 168, 176, 18, 199, 158, 167, 175, 187, 166, 174],
            'ventral': [104, 105, 106, 111, 112, 113, 114, 115, 120, 121, 122, 123, 133, 134] + [169, 177, 189, 159, 168, 176, 188, 199, 158, 167, 175, 187, 166, 174],
            'frontal': [11, 12, 18, 19, 20, 21, 25, 26, 27, 32, 33, 34, 37, 38],
            'occipital': [145,146,17,135,136,137,124,125,138,149,157,156,165]}

#signal info
pre_stim = 50 #ms before stim onset
stim_end = 300 #ms when stim goes off
post_stim = 500 #ms after stim offset
bin_length = 4 #length of each bin in ms
bin_size = 1 #how many bins were averaged over; 1 = no averaging; 5 = average over 20 ms

#calculate start window for analysis given the bin size and length
start_window = pre_stim - (bin_length*(bin_size-1)) 
#calculate the onset point of the stimulus in the dataframe given the start window and bin length
stim_onset = int(start_window/bin_length)+1 
stim_offset = int(stim_end/bin_length)+stim_onset-1
timepoints = list(range(-start_window, post_stim, bin_length)) #134 4 ms bins

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

def calc_sig_time(all_sub_decode):
    '''
    Calculate significant time
    '''
    sig_ts = []
    for time in range(0,all_sub_decode.shape[1]):
        p_val= stats.ttest_1samp(all_sub_decode[:,time], .25, axis = 0, alternative='greater')
        
        #append the p-value for every time point
        sig_ts.append(p_val[1])  

    

    #reconvert p-value list into a numpy array
    sig_ts = np.asanyarray(sig_ts)
    sig_onset = np.where(sig_ts <=.05,)[0][0]

    return sig_onset*bin_length, sig_ts

'''
load sub data
'''
onset_df = pd.DataFrame(columns = rois)
for roi in rois:
    print(roi)
    all_subs = []
    for sub in sub_list:
        #load sub data
        #data are formatted stim x time x channels
        curr_sub = np.load(f'{data_dir}/{sub}/{roi}_data.npy')

        #analyze only stim period
        curr_sub = curr_sub[:,stim_onset:,:]

        #convert to pandas dataframe
        
        all_subs.append(curr_sub)

    onset_boot = []
    for ii in range(0,iter):
        
        '''
        resample channels
        '''
        resampled_data = resample_channels(all_subs)
        '''
        decode
        '''
        all_sub_decode = []
        for sub_data in enumerate(resampled_data):
            decode_ts = decode_eeg(sub_data[1])
            all_sub_decode.append(decode_ts)
        
        all_sub_decode = np.asanyarray(all_sub_decode)
        onset, decode_sig = calc_sig_time(all_sub_decode)
        
        onset_boot.append(onset)


    onset_df[roi] = onset_boot
'''
save results
'''
onset_df.to_csv(f'{results_dir}/onsets/channel_resample_boots.csv')
#np.save(f'{results_dir}/{roi}_onset_iter{ii}.npy', onset)


