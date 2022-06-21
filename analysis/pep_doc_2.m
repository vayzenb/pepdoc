%% Preliminaries
clear
close all
clc

% Prepare the Environment
E = '/Users/clairesimmons/Desktop/Behrmann/eeglab2022.0'; %location in which EEGLab is stored
    addpath(E) %making the folder visible to MatLab  
F = '/Users/clairesimmons/Desktop/Behrmann/fieldtrip-master'; %location in which FieldTrip is stored
    addpath(F) %making the folder visible to MatLab

% add project folders
path_data = '/Users/clairesimmons/Desktop/Behrmann/Pepdoc_EEG Raw/Experiment 2/CR_raw.mff';
projectPath = 'C:\Users\clairesimmons\Desktop\Behrmann\';

%% Preprocesing

% Load data using EEGlab '%sub_raw.mff'
    eeglab; %after each major step you will need to reload the data set you just saved into EEG Lab because it's obnoxious

% Create a copy of the EEG structure to revert back to if necessary, this protects the original data from being changed
     OrigEEG=EEG;

% Plot the data
     pop_eegplot(EEG, 1, 1, 1); % check to see if the data looks good

% Rereferenced
    EEGrereferenced = pop_reref( EEG ); % pop up interactive window, rereference the average

% Lowpass [40 Hz]
    EEGlowpass = pop_firws(EEGrereferenced, 'fcutoff', 40, 'ftype', 'lowpass', 'wtype', 'hamming', 'forder', 11000, 'minphase', 0); %lowpass with 20Hz transition zone
%     pop_eegplot(EEGlowpass, 1, 1, 1);

% Downsample 
    EEGresample = pop_resample(EEGlowpass, 250); %downsample to 250 Hz (from 1000 Hz)
%     pop_eegplot(EEGresample, 1, 1, 1);
 
% Highpass
    EEGhighpass = pop_firws(EEGresample, 'fcutoff', 0.1, 'ftype', 'highpass', 'wtype', 'hamming', 'forder', 11000, 'minphase', 0); %highpass
%     pop_eegplot(EEGhighpass, 1, 1, 1);

% SAVE BEFORE ELECTRODES % check the impact of channel rejection, run ICA
% with 257
    pop_saveset(EEGhighpass);
   
%% Reject Bad Electrodes
    % this part takes a bit of intuition, which is frustrating, but the
    % first part of visualization helps to show you were you are starting

% visualize bad channels by standard deviation
    stdData = std(EEG.data,0,2); % visualize outliers
    figure; bar(stdData); 

% trim outlier is a less severe approach to bad channel rejection than
% clean_raw data, but stronger than based on kurtosis. 
   trimOutlier(EEG); % I have found this to be the best way to start

% visualize bad channels by standard deviation
    stdData = std(EEG.data,0,2); % visualize outliers again
    figure; bar(stdData);

% check kurtosis 
   pop_rejchan(EEG); % usually 0 because the outliers are still considerable

% remove less obvious bad channels using ASR
   pop_clean_rawdata(EEG); % 5, -1, 0.8, 4, -1, -1, 'availableRAM_GB', 0.05

% visually double check rejection
   pop_spectopo(EEG);

% rereference after removing bad channels from visual inspection
   pop_reref(EEG); % rereference after removing outlier
   pop_saveset(EEG); % save

% Interpolate bad channels
%      EEG_interp = pop_interp(EEGhighpass); % We did not run interpolaltion
 
%% ICA/PCA for HD EEG
    EEGica = pop_runica(EEG, 'icatype', 'runica', 'pca', 32);
    pop_saveset(EEGica);
 
% Visualize components
    EEGnocomps = pop_selectcomps(EEG); % EEG 
    EEGnocomps = pop_subcomp(EEG); % EEG without ica components, components 1 and 2 have been removed from this set
    pop_saveset(EEGnocomps); % save as %sub.set

% Add conditions to trials BEFORE rejecting artifacts [Bird: 1, Insect: 2, Graspable Object/non-tool: 3, Tool: 4]
    % change the file path for subject
    EEG = pop_importevent(EEG,'event','/Users/clairesimmons/Desktop/Behrmann/Pepdoc_EEG Raw/Experiment 2/SB_conditionfile.txt', 'fields', {'stimuli', 'luminance','exemplar'},'append', 'no');  % Add new fields to event structure

% Epoch
    EEG_epoch = pop_epoch(EEG); %, {'DIN2'}, [-.05 .5]);
    EEG_epoch = pop_rmbase (EEG_epoch); %remove baseline [-5 0]
    eeg_checkset(EEG_epoch);

% Artifact Detection
    EEGartDetect = pop_eegthresh(EEG_epoch,1,[1:EEG.nbchan],-100,100,-1.1,2.0,0,0);
    trialStruct.artThresh = [EEGartDetect.reject.rejthresh']; %1=rejectedGL_ trial
 

% Visualize the trials marked as artifacts
    EEGartDetect.reject.rejmanualE = zeros(EEGartDetect.nbchan, EEGartDetect.trials);
    EEGartDetect.reject.rejmanual = trialStruct.artThresh;
    winrej = trial2eegplot(EEGartDetect.reject.rejmanual,EEGartDetect.reject.rejmanualE,EEGartDetect.pnts,[.50, .50, .50]);
    eegplot(EEGartDetect.data,'eloc_file', EEGartDetect.chanlocs,'winrej',winrej,'srate',EEGartDetect.srate,'limits',1000*[EEGartDetect.xmin EEGartDetect.xmax], 'winlength', 4, 'spacing', 50,...
        'dispchans', EEGartDetect.nbchan);

% Reject trials with Artifacts
    EEG_rej = pop_rejepoch (EEG_epoch, trialStruct.artThresh);
    EEG = EEG_rej;
    pop_saveset(EEG_rej);

% Creating Condition Files
    % eleongated tool - tl_long
        tl_long1 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 1);
        tl_long2 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 2);
        tl_long3 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 3);
        tl_long4 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 4);
        tl_long5 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 5);
    % elongated non tool - nt_long
        nt_long1 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 1);
        nt_long2 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 2);
        nt_long3 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 3);
        nt_long4 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 4);
        nt_long5 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 5);
    %stubby tool - tl_stub
        tl_stub1 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 1);
        tl_stub2 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 2);
        tl_stub3 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 3);
        tl_stub4 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 4);
        tl_stub5 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 5);
    %stubby non tool - nt_stub
        nt_stub1 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 1);
        nt_stub2 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 2);
        nt_stub3 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 3);
        nt_stub4 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 4);
        nt_stub5 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 5);

 % Saving Condition Files % be sure to change the subjects initials

    pop_export(tl_long1, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_long/tl_long1.tsv', 'erp', 'on');
    pop_export(tl_long2, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_long/tl_long2.tsv', 'erp', 'on');
    pop_export(tl_long3, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_long/tl_long3.tsv', 'erp', 'on');
    pop_export(tl_long4, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_long/tl_long4.tsv', 'erp', 'on');
    pop_export(tl_long5, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_long/tl_long5.tsv', 'erp', 'on');

    pop_export(nt_long1, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_long/nt_long1.tsv', 'erp', 'on');
    pop_export(nt_long2, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_long/nt_long2.tsv', 'erp', 'on');
    pop_export(nt_long3, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_long/nt_long3.tsv', 'erp', 'on');
    pop_export(nt_long4, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_long/nt_long4.tsv', 'erp', 'on');
    pop_export(nt_long5, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_long/nt_long5.tsv', 'erp', 'on');

    pop_export(tl_stub1, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_stub/tl_stub1.tsv', 'erp', 'on');
    pop_export(tl_stub2, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_stub/tl_stub2.tsv', 'erp', 'on');
    pop_export(tl_stub3, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_stub/tl_stub3.tsv', 'erp', 'on');
    pop_export(tl_stub4, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_stub/tl_stub4.tsv', 'erp', 'on');
    pop_export(tl_stub5, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/tl_stub/tl_stub5.tsv', 'erp', 'on');
    
    pop_export(nt_stub1, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_stub/nt_stub1.tsv', 'erp', 'on');
    pop_export(nt_stub2, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_stub/nt_stub2.tsv', 'erp', 'on');
    pop_export(nt_stub3, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_stub/nt_stub3.tsv', 'erp', 'on');
    pop_export(nt_stub4, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_stub/nt_stub4.tsv', 'erp', 'on');
    pop_export(nt_stub5, '/Users/clairesimmons/Desktop/Behrmann/ex2_results/SB/nt_stub/nt_stub5.tsv', 'erp', 'on');