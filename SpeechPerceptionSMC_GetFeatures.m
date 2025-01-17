%% Analysis Doremi_Listen

%% 0. Set up the analysis settings

taskName = 'Doremi_listen';
if ~exist('data_tag2','var')
data_tag2 = 'ALL'; % 'ALL' 'onlyVSMC72' 'onlyVSMC64'
end
max_reaction_time = 18;


% Set conditions and labels
rest_label = 0;
iti_label = 30;
beep_label = 33;
instruction_label = 31;
task_labels = [1 2 3];
rest_attention_cue_trials = [4 7 9 13 15 18 20 23 25 27];

label_txt{1} = 'VideoAudio'; label_txt{2} = 'Video'; label_txt{3} = 'Audio';
doremi_labels{1} = 'Do'; doremi_labels{2} = 'Re'; doremi_labels{3} = 'Mi';  doremi_labels{4} = 'Fa';
doremi_labels{5} = 'So'; doremi_labels{6} = 'La'; doremi_labels{7} = 'Ti';

% Set Filter parameters
filter_params.hpfreq = 0;
filter_params.lpfreq = 0;
filter_params.notchhpfreq(1) = 49;
filter_params.notchlpfreq(1) = 51;
filter_params.notchhpfreq(2) = 97;
filter_params.notchlpfreq(2) = 103;
% filter_params.notchhpfreq(3) = 145;
% filter_params.notchlpfreq(3) = 155;

% Set Gabor parameters
p.W = 4; %spectral comp window (if Type1=Gabor, then W=#cycles in fwhm, else = width (in s) of window)
p.spectra = 1:130;
p.spectType1 = 'Gabor'; %'Mem'
p.spectType2 = 'LogPwr'; %'LogAmp' 'Amp' 'LogPwr' 'Pwr' 'Imagin' 'Complex'
p.filter_params= filter_params;

% Set classifier parameters
classifier_params.rest_label = rest_label;
classifier_params.reref_type = 'CAR'; % 'MACRO' 'CAR'
classifier_params.feature_type = 'gamma'; % 'gamma'=>(65-120Hz) 'gammaBCI'=>(65-95Hz) 'beta' 'gamma-Beta' 'BroadBand' 'BroadBandAll'
classifier_params.time_window_start = -1; %in seconds NOTE: adapted since volthe run 2. 
classifier_params.time_window_stop = +1; %in seconds
classifier_params.smooth_window = 0.05; % in seconds (0=>no smothing)
classifier_params.resample = 100; % in Hz (if =0 then no resample, if sample frequency =512 then no resample, if =2000 then down smaple, if 200 then up sample)
classifier_params.use_only_significant_electrodes = 1;
classifier_params.num_folds = 0; % # of data divisions for cross fold validation (=0 or=[] => does leave-one-out)
classifier_params.num_perms = 500;

% For each participant, the dataStructure contains information about the
% grid dimensions, layout, and included channels.
channels = dataStructure.HD_channels;
dimensions = dataStructure.HD_grid_dims;
layout = dataStructure.HD_channels_order;
   
%% 1. Load and Preprocess data

% Load data and Run the preprocessing steps
Doremi_Listen_Preprocessing;

if iscell(channels)
    chanlabels = channels;
    channels = 1:length(chanlabels);
else
    for i = 1:length(channels)
        chanlabels{i} = ['Chan', num2str(i)];
    end
end

%% 2. Extract power data and relevant features
%-- Extract the relevant features for analysis

    
    %-- Do Spectral Analysis
 
        disp(['2. Starting the Spectral Analysis for subject ', subject, '...'])
        
        if ~exist('response_spectra', 'var')
            % spectral_signal has diminsions (time X channel X frequency bin)
            switch p.spectType1
                case 'Gabor'
                    spectra_params.spectra = p.spectra;
                    spectra_params.sample_rate = data.sample_rate;
                    spectra_params.W = p.W;
                    response_spectra = jun_gabor_cov_fitted(reRef_ECoG_data',spectra_params,'',1,1); %time should be in the first dimension of the signal
            end
        end
        
        %-- Normalize the response

        disp('2. Normalizing spectral response ...');

        norm_spectra = log(abs(response_spectra).*abs(response_spectra));


        disp(['2. Computing the relevant ECoG features for subject', subject, '... ']);

        data_params.break_pnts = [];

        data_params.sample_rate = unique([data_params.sample_rate data.sample_rate]);

        total_norm_spectra = norm_spectra;
        audiodata_triggers_total = audiodata.triggers;
        data_triggers_total = data.triggers;

        clear norm_spectra;

    ECoG_feature_params = p;
    switch classifier_params.feature_type
        case 'gamma'
            ECoG_feature_params.spectra = 65:120; %65 to 120
        case 'gammaBCI'
            ECoG_feature_params.spectra = 65:95; %65 to 95
    end
    ECoG_feature_params.sample_rate = data_params.sample_rate;
    classifier_params.usable_data_buffer = (2*ECoG_feature_params.W*ceil(ECoG_feature_params.sample_rate/ECoG_feature_params.spectra(1)));
    ECoG_features_data = double(squeeze(sum(total_norm_spectra(:,:,find(p.spectra>=ECoG_feature_params.spectra(1),1,'first'):find(p.spectra<=ECoG_feature_params.spectra(end),1,'last')),3)));
    % figure; imagesc(ECoG_features_data');
    clear total_norm_spectra;
    ECoG_features_data(:,dataStructure.noisyChannels) = []; %drop noisy channels
    
    
    %-- Normalize the features data
    % Do per file norm, but not per channel
    norm_pnts = [0 data_params.break_pnts size(ECoG_features_data,1)];
    valid_spectra_pnts = [];
    for brk_pnt=1:length(norm_pnts)-1
        chunk_pnts = norm_pnts(brk_pnt)+classifier_params.usable_data_buffer+1:norm_pnts(brk_pnt+1)-classifier_params.usable_data_buffer;
        valid_spectra_pnts = [valid_spectra_pnts chunk_pnts]; %#ok<AGROW>
        data_cnk = ECoG_features_data(chunk_pnts,:);
        ECoG_features_data(norm_pnts(brk_pnt)+1:norm_pnts(brk_pnt+1),:) = ...
            (ECoG_features_data(norm_pnts(brk_pnt)+1:norm_pnts(brk_pnt+1),:)-mean(data_cnk(:)))/std(data_cnk(:));
    end; clear data_cnk chunk_pnts;
    
%% 3. Set the trial markers based on condition and sound

% The task_performance_data has information for every trial on:
%task_performance_data(trial,1) = sample when cue was presented
%task_performance_data(trial,2) = Onset Time sample of the entire trial
%task_performance_data(trial,3) = cue label of the entire trial
%task_performance_data(trial,4) = Onset Time sample for each doremi-sound individually
%task_performance_data(trial,5) = label of the doremi sound;

% The restPeriods variable has information on the when the rest trials
% occured:
% restPeriods(:,1) = onset of each rest period
% restPeriods(:,2) = offset of each rest period
% restPeriods(:,3) = restPeriod duration
% rest_to_delete =  dataStructure.rest_noisy_trials
% restPeriods(unique(rest_to_delete),:) = [];

%% 4. Smooth data and Determine which trials should stay in analysis

%*smooth gamma signal
smooth_ECoG_features_data = zeros(size(ECoG_features_data));
if round(classifier_params.smooth_window*data_params.sample_rate)>0
    for chan=1:size(ECoG_features_data,2)
        smooth_ECoG_features_data(:,chan) = smooth(ECoG_features_data(:,chan),round(classifier_params.smooth_window*data_params.sample_rate));
    end
else
    smooth_ECoG_features_data = ECoG_features_data;
end

%*resample  gamma signal
% currently no downsampling
analysis_sample_rate = data_params.sample_rate;

noisey_samples = sum(ECoG_features_data,2)>(mean(sum(ECoG_features_data,2))+(10*std(sum(ECoG_features_data,2)))) | ...
    sum(ECoG_features_data,2)<(mean(sum(ECoG_features_data,2))-(10*std(sum(ECoG_features_data,2))));

raw_trial_length = task_performance_data(2,4) - task_performance_data(1,4);% fourth column houses the start times of every doremi sound, so the raw trial length is specific for each sound

good_trials = find(task_performance_data(:,4)>0); %figure; plot(task_performance_data(:,2));
%good_trials(~ismember(task_performance_data(good_trials,1)+round(classifier_params.time_window_start*data_params.sample_rate),valid_spectra_pnts)) = []; %drop too early trials
%good_trials(~ismember(task_performance_data(good_trials,1)+raw_trial_length,valid_spectra_pnts)) = []; %drop too late trials

disp(['   Bad performance trials -> ',num2str(setdiff(1:size(task_performance_data,1),good_trials))]);

% Determine which trials are bad based on their ECoG signal
bad_ECoG_trials=[];
for gt=1:length(good_trials)
    if sum(noisey_samples(task_performance_data(good_trials(gt),1):task_performance_data(good_trials(gt),1)+raw_trial_length))>0
        bad_ECoG_trials = [bad_ECoG_trials good_trials(gt)]; %#ok<AGROW>
        good_trials(gt) = -1;
    end
end
disp(['   Bad ECoG trials -> ',num2str(bad_ECoG_trials)]);
good_trials(good_trials==-1) = [];

% extract raw trials data
raw_trial_pnts = data_params.sample_rate*classifier_params.time_window_start:round(max_reaction_time*data_params.sample_rate);
raw_trials_data = zeros(length(good_trials),length(raw_trial_pnts),size(smooth_ECoG_features_data,2));
for tr=1:length(good_trials)
    raw_trials_data(tr,:,:) = smooth_ECoG_features_data(task_performance_data(good_trials(tr),1)+raw_trial_pnts,:);
end

% Setup the data Labels
data_labels.raw_trial_pnts = raw_trial_pnts;
data_labels.sample_rate = analysis_sample_rate;
data_labels.use_only_significant_electrodes = classifier_params.use_only_significant_electrodes;
data_labels.onsets = task_performance_data(good_trials,4) - task_performance_data(good_trials,1);
% Doremi Onset Time - cue time --> delay until the sound was pronounced
data_labels.onsets(task_performance_data(good_trials,3)==rest_label) = 0;
data_labels.onsets(task_performance_data(good_trials,2) < 0) = 0; % in case of a faulty trial, do not use an onset time

doremi_data_labels = data_labels;
doremi_data_labels.labels = task_performance_data(good_trials,5);
doremi_data_labels.rest_label = 0;
doremi_data_labels.condition = task_performance_data(good_trials,3);

disp('4. Smoothed data and excluded bad trials')

%% 5. Divide data into trials and compute the means over trials

active_labels = unique(doremi_data_labels.labels); active_labels(active_labels==doremi_data_labels.rest_label) = []; 

labels_names = unique(doremi_data_labels.labels);
wndow = round(-0.5*data_params.sample_rate):round(1 * data_params.sample_rate); % take a window of 0.5 second before and 1 second after Doremi onset time
trial_strt_pnt = find(wndow > 0, 1, 'first');

nonnoisyChan = channels; nonnoisyChan(dataStructure.noisyChannels) = [];

% compute trial traces for all trials and all channels, based on the trial length specified
trial_zero_pnt = find(doremi_data_labels.raw_trial_pnts>0,1,'first');
trial_traces = zeros(length(doremi_data_labels.labels),length(wndow),size(raw_trials_data,3));
for trial=1:size(raw_trials_data,1)
    if (doremi_data_labels.onsets(trial)==0)&&((trial_zero_pnt+doremi_data_labels.onsets(trial)+wndow(1))<1)
        trial_traces(trial,:,:) = raw_trials_data(trial,1:length(wndow),:);
    else
        trial_traces(trial,:,:) = raw_trials_data(trial,trial_zero_pnt+doremi_data_labels.onsets(trial)+wndow,:); % is time period from cue time (trial_zero_pnt) + time delay until voice onset time + window we are interested in around that VOT
    end
end

% Extract the rest trials data
rest_length = min(restPeriods(:,2) - restPeriods(:,1));
rest_traces = zeros(size(restPeriods,1),rest_length,size(trial_traces,3));
for rt = 1:size(restPeriods,1)
    rest_traces(rt,:,:) =  smooth_ECoG_features_data(restPeriods(rt,1):restPeriods(rt,1)+(rest_length-1),:);
end
%save([results_directory, 'restTraces.mat'], 'rest_traces', 'nonnoisyChan');


disp('5. Divided data into trials')

% extract the period that needs to be plotted
wndow_zeroPoint = find(wndow>=0, 1, 'first');
plotWindow = [round(-0.5*analysis_sample_rate):round(1*analysis_sample_rate)];

%----- Z-score the traces based on the preceding rest trial 

trial_traces_Z = zeros(size(trial_traces));
for tr = good_trials'
    
    % define which rest period start precedes to the VOT of each trial. If
    % there is not preceding, take the one afterwards 
    [~, restToPick] =  min(abs(restPeriods(:,2) - task_performance_data(tr,1)));
    
    mean_RestZ = mean(squeeze(rest_traces(restToPick,:,:)));
    std_RestZ = std(squeeze(rest_traces(restToPick,:,:)));
    
    trial_traces_Z(tr,:,:) = (squeeze(trial_traces(good_trials == tr,:,:)) - mean_RestZ)./ std_RestZ; 
    
end

mean_trial_traces_Z = zeros(3,7,size(trial_traces_Z,2), size(trial_traces_Z,3));
for cond = 1:3
    for sound = 1:7
        
        mean_trial_traces_Z(cond,sound,:,:) = mean(trial_traces_Z(trial_labels_no_rest(:,1) == cond & trial_labels_no_rest(:,3) == sound,:,:));
        
    end
end

condition_labels = 1:3;
meanToPlot = cell(3,1);
meanToPlot_SMC = zeros(3,length(plotWindow), length(SMC_electrodes_nonnoisy));
meanToPlot_SMC_long = zeros(3, size(mean_trial_traces_Z,3), size(mean_trial_traces_Z,4));
for cond = condition_labels
    meanToPlot{cond} = squeeze(mean(squeeze(mean_trial_traces_Z(cond,:,wndow_zeroPoint+plotWindow,:))));
    meanToPlot_SMC(cond,:,:) = meanToPlot{cond}(:,ismember(nonnoisyChan, SMC_electrodes_nonnoisy));
    meanToPlot_SMC_long(cond,:,:) = squeeze(mean(squeeze(mean_trial_traces_Z(cond,:,:,:))));
end
meanToPlot_SMC_long(:,:,~ismember(nonnoisyChan, SMC_electrodes_nonnoisy)) = [];

%save([results_directory, 'meanTraces_SMC_Zscored', num2str(file), '.mat'], 'meanToPlot_SMC')
%save([results_directory, 'meanTraces_SMC_Zscored_long', num2str(file), '.mat'], 'meanToPlot_SMC_long');


%% 6. Get HFB per trial, time point, and electrode in the period of interest

% SMC_electrodes_nonnoisy = list of electrodes that were assigned to the SMC and did not have bad signal quality

trial_labels = task_performance_data(good_trials,3:5); % [condition, onset, doremi sound]

% set up the trials including rest
trial_window = round(0*analysis_sample_rate):round(0.7*analysis_sample_rate); % period around VOT

trial_traces = zeros(size(good_trials,1), length(trial_window), size(smooth_ECoG_features_data,2));
for t = 1:length(good_trials)
   trial_traces(t,:,:) = smooth_ECoG_features_data((trial_labels(t,2)+trial_window),:); 
end

%divide rest in 7 small pieces of similar size as the active trials, and extract the data
trial_traces = [trial_traces; zeros((length(restPeriods)*7),size(trial_traces,2),size(trial_traces,3))];   
trial_labels = [trial_labels; zeros((length(restPeriods)*7),3)];

counter = size(good_trials,1)+1;
for r = 1:length(restPeriods)
    
    % select starts of the 7 rest periods that all fall within the 2.5 seconds of rest, irrespective of the trial window 
    randRestStart = randi((2.5*analysis_sample_rate)-max(trial_window), [1,7]);
    
    for rp = 1:7 % 7 doremi sounds
    trial_traces(counter,:,:) = smooth_ECoG_features_data((restPeriods(r,1)+randRestStart(rp))+trial_window,:); 
    trial_labels(counter,:) = [4,(restPeriods(r,1)+randRestStart(rp)),rp]; % rest code (4), start of the rest part, and number of the rest part within this rest trial
    counter = counter+1;
    end
end


if ~exist('CAR_channels', 'var')
    CAR_channels = 1:size(elecmatrix,1);
    CAR_channels(dataStructure.noisyChannels) = [];
end

if exist('toDelete', 'var')
    CAR_channels(toDelete) = [];
end

CAR_channels_SMC = zeros(size(CAR_channels));
for i = 1:length(CAR_channels)
   CAR_channels_SMC(i) = ismember(CAR_channels(i), SMC_electrodes_nonnoisy);
end

trial_traces_SMC = trial_traces(:,:,logical(CAR_channels_SMC));

trial_traces_noRest_SMC = trial_traces_SMC(trial_labels(:,1) ~= 4,:,:);

