curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD

import sys
sys.path.insert(0,curr_dir)


import numpy as np
import pandas as pd
import pepdoc_params as params

sub_list = params.sub_list
data_dir = params.data_dir
categories = params.categories


#loop through subs
for sub in sub_list:
    #loop through categories
    for category in categories:
        #loop through each category exemplar
        for exemplar in range(1,6):
            #load tsv and convert to csv
            tsv_file = f'{data_dir}/{sub}/{category}s/{category}{exemplar}.tsv'
            csv_file = f'{data_dir}/{sub}/{category}s/{category}{exemplar}.csv'
            df = pd.read_csv(tsv_file, sep='\t')
            df.to_csv(csv_file, index=False)







