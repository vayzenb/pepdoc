clear all;
%%
%set up dirs and sub/roi params
data_dir = '/lab_data/behrmannlab/vlad/pepdoc/results_ex1';
curr_dir = '/user_data/vayzenbe/GitHub_Repos/pepdoc' 
results_dir = [curr_dir,'/results/mvgca/'];

seed_roi = 'occipital'
target_roi = 'dorsal'
control = '';
full_ts = false;

addpath('/user_data/vayzenbe/GitHub_Repos/MVGC1');
startup;

sub_list = {'AC','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ'};


if full_ts
    full_suf = '_full';
else
    full_suf = '';
end

file_suf = [control,full_suf];

cols = {'sub'}; %


%%
%start analysis loop
sn = 1; %tracks which sub num we are on
for sub = sub_list
    sub
   
   sub_summary{sn, 1} = sub{1};
   rn = 2; %rn starts at 2 because col 1 is sub

    
         
    if isempty(control)
        seed_file = [data_dir,'/',sub{1},'/',seed_roi,'_concat_ts',full_suf,'.mat'];
        target_file = [data_dir,'/',sub{1},'/',target_roi,'_concat_ts',full_suf,'.mat'];

    else
        seed_file = [data_dir,'/',sub{1},'/',seed_roi,'_',control,'_resid_ts',full_suf,'.mat'];
        target_file = [data_dir,'/',sub{1},'/',target_roi,'_',control,'_resid_ts',full_suf,'.mat'];
    end
    
    
    seed_ts = cell2mat(struct2cell(load(seed_file))); %load .mat file and convert to mat
    seed_ts = diff(seed_ts,1,1); %take diff of TS to make stationary
    seed_times = size(seed_ts); %save size for later
    
    target_ts = cell2mat(struct2cell(load(target_file))); %load .mat file and convert to mat
    target_ts = diff(target_ts,1,1); %take diff of TS to make stationary
    target_times = size(target_ts); %save size for later

    %for first sub add rois to col cell to eventually make the
    %summary columns
    if strcmp(sub{1}, sub_list{1})
        cols{end+1} = [seed_roi,'_',target_roi];
    end
    

    %determine what the min number of PCs to use
    %mvgca has to have same number of TSs
    channel_n = min([seed_times(2),target_times(2)]); 
    
    %setup empty 3D tensor and add dorsal and ventral TSs
    X = zeros(2, seed_times(1),channel_n);
    size(seed_ts)
    

    X(1,:,:)= seed_ts(:,1:channel_n);
    X(2,:,:)= target_ts(:,1:channel_n);
    
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

writetable(final_summary, [results_dir,seed_roi,'_',target_roi, file_suf,'.csv'], 'Delimiter', ',')

