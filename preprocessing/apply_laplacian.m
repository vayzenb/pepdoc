clear all;
%% 
%params

data_dir = 'C:/Users/vayze/Documents/Research/Projects/PepDoc/data/';
sub_list = {'AC','AM', 'BB','CM','CR','GG','HA','IB','JM','JR','KK','KT','MC','MH','NF','SB','SG','SOG','TL','ZZ'};
sub_list = {'GG'};
exemplar_num = 5;
cat_name = {'bird','insect','nontool','tool'};
%%

for sub_n = 1:length(sub_list)
    %clear all variables except loads
    clearvars  -except data_dir sub_n sub_list cat_name exemplar_num
    
    %import sub info
    sub_n =1 
    sub = sub_list{sub_n};
    sub_data= [data_dir,'/preprocessed/',sub,'.set'];
    importfile(sub_data)
    
    %assign data to EEG variable
    EEG.data = data;
    EEG.chanlocs = chanlocs;
    EEG.trials = trials;
    EEG.event = event;
    EEG.chanlocs = chanlocs;
    EEG.times = times;
    EEG.nbchan = nbchan;
    EEG.srate = srate;
    %EEG.epoch = epoch;
    EEG.etc = etc;
    EEG.pnts = pnts;
    EEG.setname = setname;
    %EEG.icawinv = icawinv;
    %EEG.icachansind = icachansind;
    %EEG.icaweights= icaweights;
    %EEG.icasphere= icasphere;
    EEG.xmax = xmax;
    EEG.xmin = xmin;
    %EEG.icaact = icaact (EEG.data, EEG.icaweights, 0);
    
    %EEG.urevent = urevent;
    
    
    X = [EEG.chanlocs.X];
    Y = [EEG.chanlocs.Y];
    Z = [EEG.chanlocs.Z];
    
    %apply laplacian
    lap_data = laplacian_perrinX(EEG.data,X, Y, Z);
    
    %assign lap data to EEG
    EEG.data = lap_data;
    
    %import event structures
    cond_file =[data_dir,'conditions/',sub,'_ex1/conditionfile.txt'];
    EEG = pop_importevent(EEG,'event', cond_file,'fields', {'stimuli', 'luminance','exemplar'},'append', 'no');  % Add new fields to event structure
    
    
    %loop through categories and save as TSV
    for n_cat = 1:length(cat_name)
        out_dir = [data_dir,'object_data/',sub,'/',cat_name{n_cat},'/'];
        mkdir(out_dir)
       for n_exemp = 1:exemplar_num
           outfile = [out_dir,cat_name{n_cat},int2str(n_exemp),'.tsv']
           %cat = pop_selectevent(EEG, 'stimuli', n_cat, 'exemplar', n_exemp);
           %pop_export(cat, outfile,'erp', 'on');
           
           
       end
    end
    

end
