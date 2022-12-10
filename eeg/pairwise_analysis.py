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

suf = params.suf

category_comp = [['nontool','bird','insect'],['tool','bird','insect'], ['tool','nontool','insect'], ['tool','nontool','bird']] #initial categories without tools
label_comps = [[0]*5 + [1]*5 + [2]*5]

#append each pairwise category comparison

for cat1 in range(0,len(categories)):
    for cat2 in range(cat1+1,len(categories)):
        category_comp.append([categories[cat1],categories[cat2]])
        label_comps.append([0]*5 + [1]*5)

rois = ['dorsal','ventral','occipital','frontal']


#classifier info
svm_test_size = .4
svm_splits = 100 
sss = StratifiedShuffleSplit(n_splits=svm_splits, test_size=svm_test_size)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#clf = make_pipeline(StandardScaler(), GaussianNB())

def load_data(sub_list, categories):
    
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




for comps in category_comp:
    all_sub_data = load_data(sub_list, comps)
    labels = np.repeat(np.arange(0,len(comps)), 5) #create labels for each category
    for roi in rois:
        roi_decoding = []
        roi_erp = []
        roi_sig = []
        erp_sig = []
    
        
        for sub in range(0, len(all_sub_data)):
            roi_data = select_channels(all_sub_data[sub], channels[roi])
            print('Decoding: ', sub, roi)
            
            #np.save(f'/{data_dir}/{sub_list[sub]}/{roi}_data.npy', roi_data)
            decode_results, decode_sig = decode_eeg(roi_data)
            roi_decoding.append(decode_results)
            roi_sig.append(decode_sig)

            print('ERP: ', sub, roi)
            erp_results = extract_erp(roi_data)
            roi_erp.append(erp_results)


        
        
        roi_decoding = np.asanyarray(roi_decoding)
        roi_sig = np.asanyarray(roi_sig)
        roi_erp = np.asanyarray(roi_erp)
        np.save(f'{results_dir}/decode/{roi}_decoding_{"_".join(comps)}.npy', roi_decoding)
        np.save(f'{results_dir}/decode/{roi}_decoding_sig_{"_".join(comps)}.npy', roi_sig)
        np.save(f'{results_dir}/erp/{roi}_erp_{"_".join(comps)}.npy', roi_erp)

        
