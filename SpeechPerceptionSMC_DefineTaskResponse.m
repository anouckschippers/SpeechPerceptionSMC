% Do non parametric testing of task response

%% Run for each participant: 

wndow = round(-0.5*data_params.sample_rate):round(1 * data_params.sample_rate); % take a window of 0.5 second before and 1 second after Doremi onset time
trial_strt_pnt = find(wndow > 0, 1, 'first');

period_of_interest = round(0*analysis_sample_rate):round(0.7*analysis_sample_rate);

trial_labels = task_performance_data(good_trials,:);

significantElectrodes = cell(3,1);

significance_Electrodes = zeros(size(trial_traces,3),4);
significance_Electrodes(:,1) = nonnoisyChan';

 tvals_Electrodes = zeros(size(trial_traces,3),4);
 tvals_Electrodes(:,1) = nonnoisyChan';

% for all active conditions 
for c = 1:3

    % take the mean response per active trial in this condition 
    active_trials = trial_traces(trial_labels(:,3)==c, trial_strt_pnt + period_of_interest ,:);
    
    % take the same amount of rest trials of same length, and take the mean
    % response per trial 
    rand_rest_num = randi(size(rest_traces,1), size(active_trials,1),1); % choose from the available rest trials the same number of rest trials as there are active trials
    rand_rest_timepoint = randi(size(rest_traces,2)-length(period_of_interest), length(rand_rest_num),1); % choose for each random rest trial a starting point for the period of interest, that fits in that rest period  
    
    rest_trials = zeros(size(active_trials));

    for t = 1:length(rand_rest_num)
        rest_trials(t,:,:) = rest_traces(rand_rest_num(t), rand_rest_timepoint(t) + period_of_interest, :);
    end
    
    rest_trials = squeeze(mean(rest_trials,2)); % take the mean over time
    active_trials = squeeze(mean(active_trials,2)); % take the mean over time 

    % for each electrode 

    for el = 1:size(active_trials,2)
        % do a t-test between active vs rest 

        [~, ~, ~, stats] = ttest(active_trials(:,el), rest_trials(:,el));

        true_t = stats.tstat;
        
        all_trials = [active_trials ; rest_trials];
        random_t = zeros(1000,1);
        % do 1000 random permutations of labels 
        for perm = 1:1000
            
            % get as many random trial numbers as there are active trials 
            rand_tr = randperm(size(active_trials,1)*2, size(active_trials,1));
            rand_tr_rest = 1:size(active_trials,1)*2; rand_tr_rest(rand_tr)= [];

            rand_active = all_trials(rand_tr,el); % get random trials for active
            rand_rest = all_trials(rand_tr_rest,el); % get the remaining trials for rest 

            % redo the t-test
            [~,~,~,stats] = ttest(rand_active, rand_rest);

            random_t(perm) = stats.tstat;
        end
    % get p-value as the placement of the true value in the distribution 

    t_distribution = [random_t; true_t];
    t_distribution = sort(t_distribution, 'ascend');

    significance_Electrodes(el, c+1) = (length(t_distribution) - find(t_distribution == true_t, 1, 'first')) / length(t_distribution);
    tvals_Electrodes(el, c+1) = true_t;
    end

    significantElectrodes{c} = significance_Electrodes(significance_Electrodes(:, c + 1) < (0.05/length(significance_Electrodes) ) ,1);
   

end

