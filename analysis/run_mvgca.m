clear all;
%%
%set up dirs and sub/roi params
data_dir = '/lab_data/behrmannlab/vlad/pepdoc/results_ex1';
curr_dir = '/user_data/vayzenbe/GitHub_Repos/pepdoc' 
results_dir = [curr_dir,'/results/mvgca'];
addpath('/user_data/vayzenbe/GitHub_Repos/MVGC1')
startup;

sub_list = {'AC_newepoch','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ'};


lr = {'left','right'};
dorsal_rois = {'dorsal'};
ventral_rois = {'ventral'};
cols = {'sub'}; %
file_suf = ''
%%
%start analysis loop
sn = 1; %tracks which sub num we are on
for sub = sub_list
   
   sub_summary{sn, 1} = sub{1};
   rn = 2; %rn starts at 2 because col 1 is sub

    for d_roi = dorsal_rois
         
        
        dorsal_file = [data_dir,'/',sub{1},'_',d_roi{1},'_concat_data.mat'];
        
        dorsal_ts = cell2mat(struct2cell(load(dorsal_file))); %load .mat file and convert to mat
        dorsal_times = size(dorsal_ts); %save size for later

        for v_roi = ventral_rois
            ventral_file = [data_dir,'/',sub{1},'_',v_roi{1},'_concat_data.mat'];
            ventral_ts = cell2mat(struct2cell(load(ventral_file))); %load .mat file and convert to mat
            ventral_times = size(ventral_ts); %save size for later
            
                %for first sub add rois to col cell to eventually make the
                %summary columns
                if strcmp(sub{1}, sub_list{1})
                    cols{end+1} = [d_roi{1},'_',v_roi{1}];
                end
                

                
                %determine what the min number of PCs to use
                %mvgca has to have same number of TSs
                channel_n = min([dorsal_times(2),ventral_times(2)]); 
                
                %setup empty 3D tensor and add dorsal and ventral TSs
                X = zeros(2, dorsal_times(1),channel_n);
                X(1,:,:)= dorsal_ts(:,1:);
                X(2,:,:)= ventral_ts(:,1:);
                
                
                %run mvgca
                [F, p] = mvgc_ts(X);
                f_diff = F(2,1) - F(1,2); %calculate f-diff by subtracting region predictors from eachother
                
                %add diff to cell where sn is the sub row and rn is the
                %roi col
                sub_summary{sn, rn} = f_diff;
                rn = rn +1;
                
        end
    end
    
    sn = sn + 1;
end


   

%%
%convert final summary to table and save
final_summary = cell2table(sub_summary, 'VariableNames', cols);
writetable(final_summary, [results_dir,'/mvgca_summary', file_suf,'.csv'], 'Delimiter', ',')

