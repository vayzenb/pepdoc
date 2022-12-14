clear all;
%%
%set up dirs and sub/roi params
data_dir = '/lab_data/behrmannlab/vlad/pepdoc/results_ex1';
curr_dir = '/user_data/vayzenbe/GitHub_Repos/pepdoc' 
results_dir = [curr_dir,'/results/mvgca'];
addpath('/user_data/vayzenbe/GitHub_Repos/MVGC1')
startup;

sub_list = {'AC','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ'};



dorsal_roi = 'dorsal';
ventral_roi = 'ventral';
control = '_frontal'
cols = {'sub'}; %

file_suf = control
%%
%start analysis loop
sn = 1; %tracks which sub num we are on
for sub = sub_list
    sub
   
   sub_summary{sn, 1} = sub{1};
   rn = 2; %rn starts at 2 because col 1 is sub

    
         
    if isempty(control)
        dorsal_file = [data_dir,'/',sub{1},'/',dorsal_roi,'_concat_ts.mat'];
        ventral_file = [data_dir,'/',sub{1},'/',ventral_roi,'_concat_ts.mat'];

    else
        dorsal_file = [data_dir,'/',sub{1},'/',dorsal_roi,control,'_resid_ts.mat'];
        ventral_file = [data_dir,'/',sub{1},'/',ventral_roi,control,'_resid_ts.mat'];
    end
    
    
    dorsal_ts = cell2mat(struct2cell(load(dorsal_file))); %load .mat file and convert to mat
    dorsal_ts = diff(dorsal_ts,1,1); %take diff of TS to make stationary
    dorsal_times = size(dorsal_ts); %save size for later

    
    ventral_ts = cell2mat(struct2cell(load(ventral_file))); %load .mat file and convert to mat
    ventral_ts = diff(ventral_ts,1,1); %take diff of TS to make stationary
    ventral_times = size(ventral_ts); %save size for later

    %for first sub add rois to col cell to eventually make the
    %summary columns
    if strcmp(sub{1}, sub_list{1})
        cols{end+1} = [dorsal_roi,'_',ventral_roi];
    end
    

    %determine what the min number of PCs to use
    %mvgca has to have same number of TSs
    channel_n = min([dorsal_times(2),ventral_times(2)]); 
    
    %setup empty 3D tensor and add dorsal and ventral TSs
    X = zeros(2, dorsal_times(1),channel_n);
    size(dorsal_ts)
    

    X(1,:,:)= dorsal_ts(:,1:channel_n);
    X(2,:,:)= ventral_ts(:,1:channel_n);
    
    
    %run mvgca
    [F, p] = mvgc_ts(X);
    f_diff = F(2,1) - F(1,2); %calculate f-diff by subtracting region predictors from eachother
    
    %add diff to cell where sn is the sub row and rn is the
    %roi col
    sub_summary{sn, rn} = f_diff;
    rn = rn +1;


    
    sn = sn + 1;
end


   

%%
%convert final summary to table and save
final_summary = cell2table(sub_summary, 'VariableNames', cols);
writetable(final_summary, [results_dir,'/mvgca_summary', file_suf,'.csv'], 'Delimiter', ',')

