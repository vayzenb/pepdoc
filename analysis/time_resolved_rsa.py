import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

import statsmodels.api as sm

import pdb

data_dir = f'/lab_data/behrmannlab/vlad/pepdoc/results_ex1' #read in the file; first value is the file name
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD
results_dir = f'{curr_dir}/results' #where to save the results
bin_size = 1 #20 ms bins (EACH BIN IS 4 MS SO 5 ROWS ARE 20 MS)
stim_onset = 0 #stimulus onset value (analysis time is -50, and we use 4 ms bins)
offset_window =138 #when to cut the timecourse
# bin_size = 1 
sub_list = ['AC_newepoch','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ']
categories = ['tool','nontool','bird','insect']
labels = np.asanyarray([0]*5 + [1]*5 + [2]*5 + [3]*5) #creates labels for data

rois = ['dorsal','ventral','control', 'left_dorsal', 'right_dorsal', 'left_ventral', 'right_ventral']
rois = ['dorsal','ventral']


models = ['cornet_s','skel']

model_rdms = pd.DataFrame()
for model in models:
    curr_model = pd.read_csv(f'{results_dir}/rsa/{model}_rdm.csv')
    #standardize the data
    curr_model['similarity'] = (curr_model['similarity'] - curr_model['similarity'].mean()) / curr_model['similarity'].std()

    model_rdms[model] = curr_model['similarity']



def calc_rsa(roi, model_data, covs):
    # Pull out the data
    
    X = model_data[covs]
    y = roi
    model = sm.OLS(y, X).fit()
    
    
    return model.rsquared, model.params, model.pvalues


roi_r2 = np.zeros((1,offset_window))
roi_betas = np.zeros((len(models),offset_window))
roi_ps = np.zeros((len(models),offset_window))
roi_corrs = np.zeros((len(models),offset_window))
for roi in rois:
    roi_data = np.load(f'{results_dir}/rsa/{roi}_rdm.npy')
    
    
    for time in range(0,roi_data.shape[0]):
        rdm = roi_data[time,:]

        #standardize rdms
        rdm = (rdm - rdm.mean()) / rdm.std()

        #save rdms to matrix
        r2, betas,pvals = calc_rsa(rdm, model_rdms, models)
        roi_r2[0,time] = r2 
        roi_betas[:,time] = betas
        roi_ps[:,time] = pvals


        for model in models:
            
            #calculate the correlation between the two rdms
            corr = np.corrcoef(rdm, model_rdms[model])[0,1]
            roi_corrs[models.index(model),time] = corr


    #save the data
    np.save(f'{results_dir}/rsa/{roi}_rsa_r2.npy', roi_r2)
    np.save(f'{results_dir}/rsa/{roi}_rsa_betas.npy', roi_betas)
    np.save(f'{results_dir}/rsa/{roi}_rsa_ps.npy', roi_ps)
    np.save(f'{results_dir}/rsa/{roi}_rsa_corrs.npy', roi_corrs)
