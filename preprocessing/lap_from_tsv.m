clear all;
%% 
%params

data_dir = 'C:/Users/vayze/Documents/Research/Projects/PepDoc/data/';
sub_list = {'AC','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ'};
sub_list = {'AC'};
exemplar_num = 5;
cat_name = {'bird','insect','nontool','tool'};
sub_n = 1;
cat_n = 1;
exemp_n =1;

%% 
%load sub data

sub = sub_list{sub_n};
sub_data= [data_dir,'preprocessed/',sub,'.set'];
importfile(sub_data)

clearvars  -except data_dir sub_n sub_list cat_name exemplar_num chanlocs sub sub_n cat_n exemp_n

%load tsv as table

%loop through object data
cat_file = [data_dir, 'object_data/', sub, '/',cat_name{cat_n},'s/',cat_name{cat_n},int2str(exemp_n),'.csv'];
cat_data = readtable(cat_file);
sz = size(cat_data);

%loop through and find channel indices
for ii = 1:(sz(1)-1)
    curr_channel = cat_data.Var1(ii+1,1);
    channel_inds(ii,1) = find(strcmp({chanlocs.labels}, curr_channel{1})==1);
end

chanlocs = chanlocs(channel_inds(:,1));


X = [chanlocs.X];
Y = [chanlocs.Y];
Z = [chanlocs.Z];

for cat_n = 1:length(cat_name)
   for exemp_n = 1:exemplar_num
       cat_file = [data_dir, 'object_data/', sub, '/',cat_name{cat_n},'s/',cat_name{cat_n},int2str(exemp_n),'.csv'];
       out_file = [data_dir, 'object_data/', sub, '/',cat_name{cat_n},'s/',cat_name{cat_n},int2str(exemp_n),'_lap.csv'];
       cat_data = readtable(cat_file);
        
       
       eeg_data = cat_data{2:end,2:end};
       lap_data = laplacian_perrinX(eeg_data,X, Y, Z);
       cat_data{2:end,2:end} = lap_data; %replace original category data with transformed data
       
       writetable(cat_data,out_file,'WriteVariableNames',0);
       
   end
    
end

