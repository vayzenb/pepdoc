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
path_data = 'C:\Users\clairesimmons\Desktop\Behrmann\EEG Raw\Experiment 1\TL_ex1\TL_raw.mff';
projectPath = 'C:\Users\clairesimmons\Desktop\Behrmann\';

%% Preprocesing

% Load data using EEGlab '%sub_raw.mff'
    eeglab;% after each major step you will need to reload the data set you just saved into EEG Lab because it's obnoxious

% Create a copy of the EEG structure to revert back to if necessary, this protects the original data from being changed
     OrigEEG=EEG;

% Plot the data
     pop_eegplot(EEG, 1, 1, 1); % check to see if the data looks good

% Rereferenced
    EEGrereferenced = pop_reref( EEG ); % pop up interactive window, rereference the average

% Lowpass [40 Hz]
    EEGlowpass = pop_firws(EEGrereferenced, 'fcutoff', 40, 'ftype', 'lowpass', 'wtype', 'hamming', 'forder', 11000, 'minphase', 0); %lowpass with 20Hz transition zone
    pop_eegplot(EEGlowpass, 1, 1, 1);

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
    EEG = pop_importevent(EEG,'event','/Users/clairesimmons/Desktop/Behrmann/EEG Raw/Experiment 1/JR_ex1/conditionfile.txt', 'fields', {'stimuli', 'luminance','exemplar'},'append', 'no');  % Add new fields to event structure

% Epoch
    EEG_epoch = pop_epoch(EEG); %, {'DIN2'}, [-.05 .5]);
    EEG_epoch = pop_rmbase (EEG_epoch); %remove baseline [-5 0]
    eeg_checkset(EEG_epoch);

% Artifact Detection
    EEGartDetect = pop_eegthresh(EEG_epoch,1,[1:EEG.nbchan],-100,100,-1.1,2.0,0,0);
    trialStruct.artThresh = [EEGartDetect.reject.rejthresh']; %1=rejected trial
 

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

    bird1 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 1);
    bird2 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 2);
    bird3 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 3);
    bird4 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 4);
    bird5 = pop_selectevent(EEG, 'stimuli', 1, 'exemplar', 5);

    insect1 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 1);
    insect2 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 2);
    insect3 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 3);
    insect4 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 4);
    insect5 = pop_selectevent(EEG, 'stimuli', 2, 'exemplar', 5);

    nontool1 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 1);
    nontool2 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 2);
    nontool3 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 3);
    nontool4 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 4);
    nontool5 = pop_selectevent(EEG, 'stimuli', 3, 'exemplar', 5);

    tool1 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 1);
    tool2 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 2);
    tool3 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 3);
    tool4 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 4);
    tool5 = pop_selectevent(EEG, 'stimuli', 4, 'exemplar', 5);

 % Saving Condition Files % be sure to change the subjects initials

    pop_export(bird1, '/Users/clairesimmons/Desktop/Behrmann/results/JR/birds/bird1.tsv', 'erp', 'on');
    pop_export(bird2, '/Users/clairesimmons/Desktop/Behrmann/results/JR/birds/bird2.tsv', 'erp', 'on');
    pop_export(bird3, '/Users/clairesimmons/Desktop/Behrmann/results/JR/birds/bird3.tsv', 'erp', 'on');
    pop_export(bird4, '/Users/clairesimmons/Desktop/Behrmann/results/JR/birds/bird4.tsv', 'erp', 'on');
    pop_export(bird5, '/Users/clairesimmons/Desktop/Behrmann/results/JR/birds/bird5.tsv', 'erp', 'on');

    pop_export(insect1, '/Users/clairesimmons/Desktop/Behrmann/results/JR/insects/insect1.tsv', 'erp', 'on');
    pop_export(insect2, '/Users/clairesimmons/Desktop/Behrmann/results/JR/insects/insect2.tsv', 'erp', 'on');
    pop_export(insect3, '/Users/clairesimmons/Desktop/Behrmann/results/JR/insects/insect3.tsv', 'erp', 'on');
    pop_export(insect4, '/Users/clairesimmons/Desktop/Behrmann/results/JR/insects/insect4.tsv', 'erp', 'on');
    pop_export(insect5, '/Users/clairesimmons/Desktop/Behrmann/results/JR/insects/insect5.tsv', 'erp', 'on');

    pop_export(nontool1, '/Users/clairesimmons/Desktop/Behrmann/results/JR/nontools/nontool1.tsv', 'erp', 'on');
    pop_export(nontool2, '/Users/clairesimmons/Desktop/Behrmann/results/JR/nontools/nontool2.tsv', 'erp', 'on');
    pop_export(nontool3, '/Users/clairesimmons/Desktop/Behrmann/results/JR/nontools/nontool3.tsv', 'erp', 'on');
    pop_export(nontool4, '/Users/clairesimmons/Desktop/Behrmann/results/JR/nontools/nontool4.tsv', 'erp', 'on');
    pop_export(nontool5, '/Users/clairesimmons/Desktop/Behrmann/results/JR/nontools/nontool5.tsv', 'erp', 'on');
    
    pop_export(tool1, '/Users/clairesimmons/Desktop/Behrmann/results/JR/tools/tool1.tsv', 'erp', 'on');
    pop_export(tool2, '/Users/clairesimmons/Desktop/Behrmann/results/JR/tools/tool2.tsv', 'erp', 'on');
    pop_export(tool3, '/Users/clairesimmons/Desktop/Behrmann/results/JR/tools/tool3.tsv', 'erp', 'on');
    pop_export(tool4, '/Users/clairesimmons/Desktop/Behrmann/results/JR/tools/tool4.tsv', 'erp', 'on');
    pop_export(tool5, '/Users/clairesimmons/Desktop/Behrmann/results/JR/tools/tool5.tsv', 'erp', 'on');