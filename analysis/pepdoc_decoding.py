import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pdb
import scipy.stats as stats
import scipy
import statsmodels.api as sm

data_dir = f'/lab_data/behrmannlab/claire/pepdoc/results_ex1' #read in the file; first value is the file name
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
results_dir = f'{curr_dir}/results' #where to save the results
bin_size = 5 #20 ms bins (EACH BIN IS 4 MS SO 5 ROWS ARE 20 MS)
# bin_size = 1 
categories = ['tool','nontool','bird','insect']
labels = np.asanyarray([0]*5 + [1]*5 + [2]*5 + [3]*5) #creates labels for data

#sub codes
sub_list = ['AC_newepoch','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ']

#channels
left_dorsal_channels = [77, 78, 79, 80, 86, 87, 88, 89, 98, 99, 100, 110, 109, 118] 
right_dorsal_channels =  [131, 143, 154, 163, 130, 142, 153, 162, 129, 141, 152, 128, 140, 127]
dorsal_channels = left_dorsal_channels + right_dorsal_channels

left_ventral_channels = [104, 105, 106, 111, 112, 113, 114, 115, 120, 121, 122, 123, 133, 134]
right_ventral_channels = [169, 177, 189, 159, 168, 176, 18, 199, 158, 167, 175, 187, 166, 174]
ventral_channels = left_ventral_channels + right_ventral_channels

control_channels =  [11, 12, 18, 19, 20, 21, 25, 26, 27, 32, 33, 34, 37, 38]

#classifier info
svm_test_size = .4
svm_splits = 50
sss = StratifiedShuffleSplit(n_splits=svm_splits, test_size=svm_test_size)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

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
    decode_sig = decode_sig[10:]
    onset = np.where(decode_sig <= .05)[0][0]


    #decode_sig = np.asanyarray(decode_sig)
    cat_decode = np.asanyarray(cat_decode)

    return cat_decode

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


all_sub_data = load_data(sub_list)

left_dorsal_results = []
right_dorsal_results = []
left_ventral_results = []
right_ventral_results = []
dorsal_results = []
ventral_results = []
control_results = []

for sub in range(0, len(all_sub_data)):
    
    left_dorsal_data = select_channels(all_sub_data[sub], left_dorsal_channels)
    left_dorsal_results.append(decode_eeg(left_dorsal_data))
    np.save(f'{results_dir}/left_dorsal_decoding.npy', left_dorsal_results)

    right_dorsal_data = select_channels(all_sub_data[sub], right_dorsal_channels)
    right_dorsal_results.append(decode_eeg(right_dorsal_data))
    np.save(f'{results_dir}/right_dorsal_decoding.npy', right_dorsal_results)

    left_ventral_data = select_channels(all_sub_data[sub], left_ventral_channels)
    left_ventral_results.append(decode_eeg(left_ventral_data))
    np.save(f'{results_dir}/left_ventral_decoding.npy', left_ventral_results)

    right_ventral_data = select_channels(all_sub_data[sub], right_ventral_channels)
    right_ventral_results.append(decode_eeg(right_ventral_data))
    np.save(f'{results_dir}/right_ventral_decoding.npy', right_ventral_results)

    
    dorsal_data = select_channels(all_sub_data[sub], dorsal_channels)
    dorsal_results.append(decode_eeg(dorsal_data))
    np.save(f'{results_dir}/dorsal_decoding.npy', dorsal_results)

    ventral_data = select_channels(all_sub_data[sub], ventral_channels)
    ventral_results.append(decode_eeg(ventral_data))
    np.save(f'{results_dir}/ventral_decoding.npy', ventral_results)

    control_data = select_channels(all_sub_data[sub], control_channels)
    control_results.append(decode_eeg(control_data))
    np.save(f'{results_dir}/control_decoding.npy', control_results)
