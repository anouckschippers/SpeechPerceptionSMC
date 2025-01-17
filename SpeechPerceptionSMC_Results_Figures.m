
redblueCmap = redblue(100);

%% Load the data
% for all participants (S) the following information is loaded and structured:

% tvals(S).p_values = significance_Electrodes; 
% tvals(S).significantElectrodes = significantElectrodes; 
% tvals(S).tVals = tvals_Electrodes; 

% % Smooth the feature response with a 0.05 s window
% smooth_ECoG_features_data = zeros(size(ECoG_features_data));
% for chan=1:size(ECoG_features_data,2)
%     smooth_ECoG_features_data(:,chan) = smooth(ECoG_features_data(:,chan),round(0.05*500));
% end
% clear ECoG_features_data; clear ECoG_feature_params; clear data; clear data_params; clear audiodata; clear valid_spectra_pnts;
% 
% allData(S).ECoG_features = smooth_ECoG_features_data; 
% allData(S).task_performance_data = task_performance_data; 
% allData(S).avgTrialTraces = meanToPlot_SMC_long; 
% allData(S).trialTraces = trial_traces_noRest_SMC; 
% allData(S).trialLabels = trial_labels_no_rest; 
% restOnsets{S} = restPeriods; 

% ElectrodeCoordinates{S} = list of electrode coordinates in MNI

% MNI_brain = cortex in MNI space to plot results over subjects on

% SMC_electrodes_nonnoisy{S] = list of electrodes that were placed over the SMC and were not excluded due to poor signal quality


%% Figure 1A: plot the electrode coordinates on the native space brains
elCols = [0 0 1; 0 0 1; 0 1 0 ]; % change the color per participant

for s = 1:3

    % Load the cortex reconstruction (cortex)
    % Load the electrode coordinates in native space (elecmatrix)

    figure; ctmr_gauss_plot( cortex,[0 0 0],0); view(270, 0); light;
    for i = 1:length(elecmatrix)
        hold on;
        if ismember(i, SMC_electrodes_nonnoisy{s})
            elcol = elCols(s,:); % change the color per participant
        else
            elcol = [0.5 0.5 0.5];
        end

        plotSpheres(elecmatrix(i,:), elcol);

    end

end

%% Figure 1B: Plot the electrodes on the brain, highlighting which are SMC

[s_x,s_y,s_z] = sphere(100);
el_size = 0.6;
x_coord_shift = -15;
z_coord_shift = 0;
y_coord_shift = 0;

h = figure; hold on;
tripatch(MNI_brain.cortex,h,zeros(size(MNI_brain.cortex.vert,1),1));
set(gcf,'renderer','zbuffer');
colormap(cm); shading interp;
a=get(gca);
d=a.CLim;
set(gca,'CLim',[-max(abs(d)) max(abs(d))]);
material dull; %shiny dull metal
view(MNI_view_possition);
l=light; lighting gouraud; %gouraud; phong flat
set(l,'Position',MNI_light_possition);
%set(gca,'clim',[-1 2]);
freezeColors; axis off;

for s = 1:3 % subject

    for el=1:size(ElectrodeCoordinates{s},1)

        if ismember(el, SMC_electrodes_nonnoisy{s}) % only the included SMC electrodes
            if s == 1
                electrode_color = [0 0 1];
                el_size = 0.6;

            elseif s == 2
                electrode_color = [0 1 0];
                el_size = 0.7;

            elseif s == 3
                electrode_color = [1 0 0];
                el_size = 0.6;

            end

            h_s = surf((s_x*el_size)+ElectrodeCoordinates{s}(el,1)+x_coord_shift,(s_y*el_size)+ElectrodeCoordinates{s}(el,2)+y_coord_shift, (s_z*el_size)+ElectrodeCoordinates{s}(el,3)+z_coord_shift);
            h_s.EdgeAlpha = 0;

            h_s.FaceColor = electrode_color;

        end
    end
end

%% Fig 2A: Plot the t-values on the brain

% find the absolute highest t-value per subject, irrespective of condition,
% which will later be used to color the electrodes
lims_t_posneg = [-inf -inf -inf];
for s = 1:3
    for c = 1:3
        if max(abs(tvals(s).tVals(:,c+1))) > lims_t_posneg(s)
            lims_t_posneg(s) = max(abs(tvals(s).tVals(:,c+1)));
        end
    end
end

% plot for each condition the electrodes on the cortex
conditionLabel{1} = 'Audiovisual'; conditionLabel{2} = 'Visual'; conditionLabel{3} = 'Auditory';

cmap = redblue(200);

[s_x,s_y,s_z] = sphere(100);
el_size = 0.6;
x_coord_shift = -15;
z_coord_shift = 0;
y_coord_shift = 0;

for s = 1:3
    for c = 1:3
        h=figure; hold on;
        tripatch(MNI_brain.cortex,h,zeros(size(MNI_brain.cortex.vert,1),1));
        set(gcf,'renderer','zbuffer');
        colormap(cm); shading interp;
        a=get(gca);
        d=a.CLim;
        set(gca,'CLim',[-max(abs(d)) max(abs(d))])

        material dull; %shiny dull metal
        title(['S' num2str(s), ' t-values in condition ', conditionLabel{c}]);
        view(MNI_view_possition);
        l=light; lighting gouraud; %gouraud; phong flat
        set(l,'Position',MNI_light_possition);
        %set(gca,'clim',[-1 2]);
        freezeColors; axis off;

        values = tvals(s).tVals(:,c+1);
        % rescale the values to values between -1 and 1, where 1 is the the maximum absolute value so we can plot their
        % respective colors
        values_rescaled = (values - (lims_t_posneg(s)*-1)) ./ (lims_t_posneg(s) - (lims_t_posneg(s)*-1));
        values_rescaled = 2 * values_rescaled - 1;

        for el=1:size(ElectrodeCoordinates{s},1)

            if ismember(el, SMC_electrodes_nonnoisy{s})
                idx = find(tvals(s).tVals(:,1) == el);

                % color the electrode based on their t-value
                value_idx = round(values_rescaled(idx)*100);
                if value_idx ~= 0
                    electrode_color = cmap(100 + value_idx,:); % pos values will be red, neg values will be blue
                else % value 0, color white
                    electrode_color = cmap(100,:);
                end

                % scale the electrode size based on their t-value
                el_size = 0.6+ abs(values_rescaled(idx));

                % to mark significant electrodes 
                if ismember(el, tvals(s).significantElectrodes{c})
                    h_s = surf((s_x*(el_size+0.3))+ElectrodeCoordinates{s}(el,1)+x_coord_shift+3,(s_y*(el_size+0.3))+ElectrodeCoordinates{s}(el,2), (s_z*(el_size+0.3))+ElectrodeCoordinates{s}(el,3), 'FaceLighting', 'none', 'EdgeLighting','none', 'EdgeAlpha', 0, 'FaceColor', [0 0 0]);
                end

                h_s = surf((s_x*el_size)+ElectrodeCoordinates{s}(el,1)+x_coord_shift,(s_y*el_size)+ElectrodeCoordinates{s}(el,2), (s_z*el_size)+ElectrodeCoordinates{s}(el,3), 'FaceLighting', 'none', 'EdgeLighting','none', 'EdgeAlpha', 0);

                h_s.FaceColor = electrode_color;

            end
        end
    end
end


%% Fig 2B: Plot the traces over time in the significant electrodes, for all conditions
% all electrodes that are significant in ALL conditions.

wndow = round(-0.5*500):500; % take a window of 0.5 second before and 1 second after Doremi onset time
class_window = 0:round(0.7*500); % period after VOT: 700 ms
period_to_indicate = find(wndow == 0, 1, 'first') + class_window;

plotcolors = [0 0 0; 0 0 1; 1 0 0];

figure;
for s = 1:3
    subplot(3,1,s)

    allSign = intersect(tvals(s).significantElectrodes{1}, tvals(s).significantElectrodes{2}); allSign = intersect(allSign, tvals(s).significantElectrodes{3});
    signElecs = unique(allSign);

    % take only the SMC electrodes that were not noisy
    signElecs(~ismember(signElecs, SMC_electrodes_nonnoisy{s})) = [];

    ylim([-0.2 1.4])
    ylims = ylim;
    patch([period_to_indicate fliplr(period_to_indicate)], [ones(1,length(period_to_indicate))*ylims(1) fliplr(ones(1,length(period_to_indicate))*ylims(2))], [0.7 0.7 0.7], 'LineStyle', 'none', 'EdgeColor', [0.7 0.7 0.7])
    hold on;
    xline(find(wndow==0, 1, 'first'), '--k')

    for c  = 1:3
        plot(mean(allData(s).avgTrialTraces(c,:,ismember(SMC_electrodes_nonnoisy{s},signElecs)),3), 'Color', plotcolors(c,:), 'LineWidth', 3);
        hold on;
    end

    xticks(1:round(0.25*500):length(wndow))
    xticklabels(round(wndow(1)/500,2):0.25:round(wndow(end)/500,2))
    title(['Subject ',  num2str(s)])
end
%legend('Time window of interest','Onset' ,'Audiovisual', 'Video-Only', 'Audio-only')

%% Figure 2C: Make boxplots of response per condition and test for significance
% electrodes that are significant in ALL conditions

for s = 1:3

    %take only the SMC nonnoisy channels
    ch = ismember(tvals(s).tVals(:,1),SMC_electrodes_nonnoisy{s});

    rest_length = min(restOnsets{s}(:,2) - restOnsets{s}(:,1));
    rest_traces = zeros(size(restOnsets{s},1),rest_length,sum(ch));
    for rt = 1:size(restOnsets{s},1)
        rest_traces(rt,:,:) =  allData(s).ECoG_features(restOnsets{s}(rt,1):restOnsets{s}(rt,1)+(rest_length-1),ch);
    end

    % normalize the trials based on the preceding rest period
    trial_traces_Z = zeros(size(allData(s).trialTraces));
    for tr = 1:length(allData(s).trialLabels)

        % define which rest period start is closest to the VOT of each trial
        [~, restToPick] =  min(abs(restOnsets{s}(:,2) - allData(s).task_performance_data(allData(s).task_performance_data(:,4)==allData(s).trialLabels(tr,2),1)));

        mean_RestZ = mean(squeeze(rest_traces(restToPick,:,:)));
        std_RestZ = std(squeeze(rest_traces(restToPick,:,:)));

        trial_traces_Z(tr,:,:) = (squeeze(allData(s).trialTraces(tr,:,:)) - mean_RestZ)./ std_RestZ;

        allData(s).trialTraces_Z = trial_traces_Z;

    end

    allSign = intersect(tvals(s).significantElectrodes{1}, tvals(s).significantElectrodes{2}); allSign = intersect(allSign, tvals(s).significantElectrodes{3});
    signElecs = unique(allSign);
    signElecs(~ismember(signElecs, SMC_electrodes_nonnoisy{s})) = [];

    mean_trialResp = mean(squeeze(mean(trial_traces_Z(:,:,ismember(SMC_electrodes_nonnoisy{s}, signElecs)),3)),2);
    trialLabels = allData(s).trialLabels(:,1);

    [~, ~, stats] = anova1(mean_trialResp, trialLabels) ;
    multcompare(stats)
    figure; boxplot(mean_trialResp, trialLabels); ylim([-1.2 2.2]);

end


%% Figure 3: Interpolate the R2s on the MNI brain

%Make sure all electrode interpolations are plotted outside of the brain
%ElecCoordinates_projected = every electrode, projected on the MNI cortex
%surface. 

brain = MNI_brain.cortex.vert;
cmap = jet(100);
cmap(1,:) = [0.5 0.5 0.5];

max_t = [-inf -inf -inf];
min_t = [inf inf inf];

% find the POSITIVE min and max per subject, over all conditions
for s = 1:3
    for c = 1:3

        % Get for this condition the positive values
        vals_pos = tvals(s).tVals(:,c+1);
        vals_pos(vals_pos < 0) = [];

        if max(vals_pos) > max_t(s)
            max_t(s) = max(vals_pos);
        end

        if min(vals_pos) < min_t(s)
            min_t(s) = min(vals_pos);
        end

    end
end


for c = 1:3 % condition

    gsp = 4; % gaussian size
    sub_all = zeros(length(MNI_brain.cortex.vert(:,1)),1);

    for s = 1:3 %subjects
        sub_c = zeros(length(MNI_brain.cortex.vert(:,1)),1);

        % normalize the R2 values between min and max value
        weights = (tvals(s).tVals(:,c+1) - min_t(s)) ./ (max_t(s) - min_t(s)); % take the R2 values of this subject in this condition

        for e = 1:length(tvals(s).tVals) % for all the electrodes we have R2 value of for this subject and condition
            if ismember(tvals(s).tVals(e,1), SMC_electrodes_nonnoisy{s})
                if tvals(s).tVals(e,c+1) >= 0 % only if the t-value is positive

                    b_z = abs(brain(:,3) - ElecCoordinates_projected{s}(tvals(s).tVals(e,1),3));
                    b_y = abs(brain(:,2) - ElecCoordinates_projected{s}(tvals(s).tVals(e,1),2));
                    b_x = abs(brain(:,1) - ElecCoordinates_projected{s}(tvals(s).tVals(e,1),1));
                    d = weights(e) * exp((-(b_x.^2+b_z.^2+b_y.^2).^.5) /gsp^.5); %exponential fall off
                    sub_c = sub_c + d; % add the shade to the previous shades

                end
            end
        end

        sub_all = sub_all + sub_c;

    end

    sub_all(round(sub_all,2) <= 0) = nan;

    % find the center
    sub_center = round(sub_all);
    sub_center(sub_center <= 0) = nan;
    sub_center_dors = sub_center;
    sub_center_dors(MNI_brain.cortex.vert(:,3) < 30) = nan;
    [~,max_dors] = max(sub_center_dors);

    disp('MNI coord dorsal:')
    MNI_brain.cortex.vert(max_dors,:)

    sub_center_ventr  = sub_center;
    sub_center_ventr(MNI_brain.cortex.vert(:,3) > 30) = nan;
    [~, max_vent] = max(sub_center_ventr);
    disp('MNI coord ventral:')
    MNI_brain.cortex.vert(max_vent,:)

    % plot
    h = figure; hold on;
    %h = tripatch(MNI_brain.cortex,h,zeros(size(MNI_brain.cortex.vert,1),1));
    shading interp;
    a=get(gca);
    d=a.CLim;
    set(gca,'CLim',[-max(abs(d)) max(abs(d))]);
    colormap(cm);

    l=light;
    %colormap(bone); %colormap(bone);
    lighting gouraud; %gouraud; phong flat
    %material dull; %shiny dull metal
    material([.3 .8 .1 10 1]);
    view(MNI_view_possition);
    set(l,'Position',MNI_light_possition);

    axis off;
    set(gcf,'renderer','zbuffer');

    freezeColors

    h1 = tripatch(MNI_brain.cortex, h, sub_all); hold on;
    colormap(cmap); set(gca, 'clim', [0 1.5])
    %set(gca, 'XLim', xlims); set(gca, 'YLim', ylims); set(gca, 'ZLim', zlims);
    material dull; %shiny dull metal
    shading interp;
    title(['Interpolated Tval for all subjects in condition ', conditionLabel{c}])
    set(gcf,'renderer','Painters')
end

%% Figure 4A: Select the top 20 % of data

best_percent = 20;

% select the top 20 percent of electrodes as the Electrodes of Interest

EOI = cell(3,3);
tvals_bestSelected{s,c}= cell(3,3);
ElecCoordinates_bestSelected = cell(3,3);

[s_x,s_y,s_z] = sphere(100);
el_size = 1.3;
x_coord_shift = -15;

tvals_sorted = cell(3,3);
for s = 1:3
    for c = 1:3
        t = [tvals(s).tVals(:,1), tvals(s).tVals(:, c+1)];
        tvals_sorted{s,c} = sortrows(t, 2, 'descend');
    end
end

conditionLabel{1} = 'Audiovisual'; conditionLabel{2} = 'Visual'; conditionLabel{3} = 'Audio';
for c = 1:3

    h=figure; hold on;
    tripatch(MNI_brain.cortex,h,zeros(size(MNI_brain.cortex.vert,1),1));
    set(gcf,'renderer','zbuffer');
    colormap(cm); shading interp;
    a=get(gca);
    d=a.CLim;
    set(gca,'CLim',[-max(abs(d)) max(abs(d))])
    material dull; %shiny dull metal
    title(['Top 20% responsive electrodes per Sub in condition ', conditionLabel{c}]);
    view(MNI_view_possition);
    l=light; lighting gouraud; %gouraud; phong flat
    set(l,'Position',MNI_light_possition);
    %set(gca,'clim',[-1 2]);
    freezeColors; axis off;

    for s = 1:3

        numToTake = ceil(length(SMC_electrodes_nonnoisy{s})/(100/best_percent));

        % ignore the non-SMC ones
        tvals_sorted{s,c}(~ismember(tvals_sorted{s,c}(:,1), SMC_electrodes_nonnoisy{s}),:) = [];

        EOI{s,c} = tvals_sorted{s,c}(1:numToTake,1);

        tvals_bestSelected{s,c} = tvals_sorted{s,c}(1:numToTake,:);

        ElecCoordinates_bestSelected{s,c} = ElectrodeCoordinates{s,1}(EOI{s,c},:);

        for el=1:size(ElecCoordinates_bestSelected{s,c},1)

            if s == 1
                electrode_color = [0 0 1];
                el_size = 0.6;
            elseif s == 2
                electrode_color = [0 1 0];
                el_size = 0.7;
            elseif s == 3
                electrode_color = [1 0 0];
                el_size = 0.6;
            end


            h_s = surf((s_x*el_size)+ElecCoordinates_bestSelected{s,c}(el,1)+x_coord_shift,(s_y*el_size)+ElecCoordinates_bestSelected{s,c}(el,2), (s_z*el_size)+ElecCoordinates_bestSelected{s,c}(el,3), 'FaceLighting', 'none', 'EdgeLighting','none');
            h_s.EdgeAlpha = 0;

            h_s.FaceColor = electrode_color;

        end
    end

end

%% Read the C-Grid coordinates

% cGrid = cGrid cortical representation of S1 
% cGrid_coordinates = list of Cgrid x and y coordinates for every subject 

%% Fig 4B: Cluster the top electrodes of all subjects combined and determine the significance

allCGrid = [CGrid_coordinates{1,1}; CGrid_coordinates{2,1}; CGrid_coordinates{3,1}];
ClusterCenters_x = cell(1,3);
ClusterCenters_y = cell(1,3);
Clusters_X = cell(1,3);
Clusters_Y = cell(1,3);
elecsToTake_perCond = cell(3,2);

totCGrid_coords = [CGrid_coordinates{1,1}(SMC_electrodes_nonnoisy{1,1},:); CGrid_coordinates{2,1}(SMC_electrodes_nonnoisy{2,1},:); CGrid_coordinates{3,1}(SMC_electrodes_nonnoisy{3,1},:)];

% define the ranges for the electrode coordinates for consistent
% plotting
xmin = min(totCGrid_coords(~isnan(totCGrid_coords(:,1)),1));
xmax = max(totCGrid_coords(~isnan(totCGrid_coords(:,1)),1));
ymin = min(totCGrid_coords(~isnan(totCGrid_coords(:,2)),2));
ymax = max(totCGrid_coords(~isnan(totCGrid_coords(:,2)),2));

for c = 1:3 % condition

    elecsToTake_all = [];
    elecsToTake_origin = [];
    for i = 1:3 % subject

        % Select the C-grid coordinates for the EOI

        elecsToTake = CGrid_coordinates{i,1}(EOI{i,c},:);
        elecsToTake_all = [elecsToTake_all; elecsToTake]; %#ok<AGROW>
        elecsToTake_origin = [elecsToTake_origin; ones(length(elecsToTake),1)*i EOI{i,c}]; %#ok<AGROW>

    end
    elecsToTake_perCond{c,1} = elecsToTake_all;
    elecsToTake_perCond{c,2} = elecsToTake_origin;

    % Do the clustering on the x-coordinate
    [idx, centr] = kmeans(elecsToTake_all(:,1),2, 'Display', 'final');

    Clusters_X{1,c} = idx;
    ClusterCenters_x{1,c} = centr;
    ClusterDistances_x = abs(diff(centr));

    % plot the clusters with their centers
    figure;
    subplot(2,2,1);
    imagesc(flipud(squeeze(cGrid(:,:,1)))'); set(gca, 'YDir', 'normal'); colormap(gray);
    hold on;

    for i = 1:length(allCGrid)
        plot(allCGrid(i,1), allCGrid(i,2), 'Color', [0.2 0.2 1], 'MarkerSize', 10, 'Marker', '.');
    end

    for el = 1:length(idx)
        if idx(el) == 1
            elCol = [1 0.2 1];
        elseif idx(el) == 2
            elCol = [0.2 1 1];
        else
            elCol = [0 0 0];
        end
        plot(elecsToTake_all(el,1), elecsToTake_all(el,2), 'Color', elCol, 'Marker', '.', 'MarkerSize', 12);
        hold on;
    end
    plot(centr(1), 32, '+k', 'MarkerSize', 15);
    plot(centr(2), 32, '+k', 'MarkerSize', 15);
    plot([centr(1) centr(2)], [32 32], '--k')
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);

    % compute 1000 clusters based on random top electrodes

    RandElec_all = [];
    RandCoord_all = [];

    %         totNumRand = length(EOI{1,j}) + length(EOI{2,j}) + length(EOI{3,j});
    %         totNonNoisy = length(SMC_electrodes_nonnoisy{1,1}) + length(SMC_electrodes_nonnoisy{2,1}) + length(SMC_electrodes_nonnoisy{3,1});
    %
    for k = 1:1000

        rand_coords = [];
        for s = 1:3 % get random coordinates from each subject

            whichRands = randperm(length(SMC_electrodes_nonnoisy{s,1}), length(EOI{s,c}));
            subcoords = CGrid_coordinates{s,1}(SMC_electrodes_nonnoisy{s,1},:);
            rand_coords = [rand_coords ; subcoords(whichRands,:)]; %#ok<AGROW>
        end

        [~, centrR] = kmeans(rand_coords(:,1),2);
        randCenters_x{i,k} = centrR;

        randClusterDistances_x(i,k,1) = abs(diff(centrR));

        [~, centrR] = kmeans(rand_coords(:,2),2);
        randCenters_y{i,k} = centrR;

        randClusterDistances_y(i,k,1) = abs(diff(centrR));

    end

    %>> For x
    % Create the distribution of the 100 random clusters in their center
    % distances: take the absolute distance between the x and y coordinates
    randDistances_hor = sort(abs(squeeze(randClusterDistances_x(i,:,1))));
    sign_hor = 1 - (find(randDistances_hor >= abs(ClusterDistances_x), 1, 'first')/1000);

    title(['Top 20% Electrode X-Coordinate Clusters, p-val = ',  num2str(sign_hor)])


    %>>>>>>>>>> Repeat for the Y coordinate
    [idx, centr] = kmeans(elecsToTake_all(:,2),2, 'Display', 'final');

    Clusters_Y{1,c} = idx;
    ClusterCenters_y{1,c} = centr;
    ClusterDistances_y = abs(diff(centr));

    % plot the clusters with their centers
    subplot(2,2,3);
    imagesc(flipud(squeeze(cGrid(:,:,1)))'); set(gca, 'YDir', 'normal'); colormap(gray);
    hold on;
    for i = 1:length(allCGrid)
        plot(allCGrid(i,1), allCGrid(i,2), 'Color', [0.2 0.2 1], 'MarkerSize', 10, 'Marker', '.');
    end

    for el = 1:length(idx)
        if idx(el) == 1
            elCol = [1 0.2 1];
        elseif idx(el) == 2
            elCol = [0.2 1 1];
        else
            elCol = [0 0 0];
        end
        plot(elecsToTake_all(el,1), elecsToTake_all(el,2), 'Color', elCol, 'Marker', '.', 'MarkerSize', 12);
        hold on;
    end
    plot(22, centr(1), '+k', 'MarkerSize', 15);
    plot(22, centr(2), '+k', 'MarkerSize', 15);
    plot([22 22],[centr(1) centr(2)] , '--k')
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);

    randDistances_vert = sort(abs(squeeze(randClusterDistances_y(i,:,1))));
    sign_vert = 1 - (find(randDistances_vert >= abs(ClusterDistances_y), 1, 'first')/1000);

    title(['Top 20% Electrode Y-Coordinate Clusters, p-val = ', num2str(sign_vert)])


    true_elecs = nan(length(totCGrid_coords),2);
    true_elecs(1:length(elecsToTake_all),:) = elecsToTake_all;

    subplot(2,2,2);
    histogram((totCGrid_coords(~isnan(totCGrid_coords(:,1)),1)), 'Normalization', 'probability'); hold on;
    histogram(true_elecs(:,1), 'FaceColor', 'r', 'Normalization', 'probability')
    legend('Random', 'Best')
    title('Electrode X-Coordinate distribution')
    xlim([xmin, xmax]);

    subplot(2,2,4);
    histogram((totCGrid_coords(~isnan(totCGrid_coords(:,2)),2)), 'Orientation', 'horizontal', 'Normalization', 'probability'); hold on;
    histogram(true_elecs(:,2), 'FaceColor', 'r', 'Orientation', 'horizontal', 'Normalization', 'probability');
    legend('Random', 'Best')
    title('Electrode Y-Coordinate distribution')
    ylim([ymin, ymax]);

    sgtitle(['All Subjects, Condition ', conditionLabel{c}])

    set(gcf,'renderer','Painters')

    figure;
    subplot(2,1,1);
    histogram(randDistances_hor,100);
    hold on; xline(abs(ClusterDistances_x), 'r');
    title(['Difference in Anterior-Posterior centers, p-value True Distance = ', num2str(sign_hor)]);

    subplot(2,1,2);
    histogram(randDistances_vert,100);
    hold on; xline(abs(ClusterDistances_y), 'r');
    title(['Difference in Dorsal-Ventral centers, p-value True Distance = ', num2str(sign_vert)]);

    set(gcf,'renderer','Painters')

end


%% Get the traces over time in the Dorsal-Ventral clustered electrodes

% extract the trial traces that correspond to each EOI and their cluster

cluster_one_traces = cell(3,3);
cluster_two_traces = cell(3,3);

cluster_one_elecs = cell(3,3);
cluster_two_elecs = cell(3,3);

elecsToTake_all_Clust1 = zeros(1,2);
elecsToTake_all_Clust2 = zeros(1,2);

elecs_per_cond_sub_Clust1 = cell(3,3);
elecs_per_cond_sub_Clust2 = cell(3,3);

for cond = 1:3
    [m, idx_max] = max(ClusterCenters_y{cond}); % select the dorsal cluster by taking the cluster with the highest y value centroid

    for el = 1:length(elecsToTake_perCond{cond,1}) %

        origSub = elecsToTake_perCond{cond,2}(el,1);
        origElec = elecsToTake_perCond{cond,2}(el,2);

        if Clusters_Y{cond}(el) == idx_max
            elecs_per_cond_sub_Clust1{origSub,cond} = [elecs_per_cond_sub_Clust1{origSub,cond}; origElec];

            % if this electrode is not already included in the list of
            % electrodes for this cluster, irrespective of condition
            if sum(ismember(elecsToTake_all_Clust1(:,1), origSub) .* ismember(elecsToTake_all_Clust1(:,2), origElec)) == 0

                elecsToTake_all_Clust1 = [elecsToTake_all_Clust1; origSub, origElec]; %#ok<AGROW>


                for c = 1:3 % condition

                    cluster_one_traces{origSub,c} = [cluster_one_traces{origSub,c}; allData(s).avgTrialTraces(c,:,SMC_electrodes_nonnoisy{origSub}==origElec)]   ;
                    cluster_one_elecs{origSub,c} = [cluster_one_elecs{origSub,c}; origElec];

                end

            end

        else
            % if this electrode is not already included in the list of
            % electrodes for this cluster, irrespective of condition
            elecs_per_cond_sub_Clust2{origSub,cond} = [elecs_per_cond_sub_Clust2{origSub,cond}; origElec];

            if sum(ismember(elecsToTake_all_Clust2(:,1), origSub) .* ismember(elecsToTake_all_Clust2(:,2), origElec)) == 0

                elecsToTake_all_Clust2 = [elecsToTake_all_Clust2; origSub, origElec]; %#ok<AGROW>

                for c = 1:3
                    cluster_two_traces{origSub,c} = [cluster_two_traces{origSub, c}; allData(s).avgTrialTraces(c,:,SMC_electrodes_nonnoisy{origSub}==origElec)];
                    cluster_two_elecs{origSub,c} = [cluster_two_elecs{origSub,c}; origElec];
                end
            end
        end


    end


end
% delete the first row because these were initialized as zeroes
elecsToTake_all_Clust1(1,:) = [];
elecsToTake_all_Clust2(1,:) = [];

tot_el_clusterOne = length(elecsToTake_all_Clust1);
tot_el_clusterTwo = length(elecsToTake_all_Clust2);

%% Get the trial information within the eletrodes within a cluster

Trials_perSubCondClust = cell(3,3,2);
Trials_perSubCondClust_notAvg = cell(3,3,2);
Trials_perSubCondClust_onlyAVelecs = cell(3,3,2);
Trials_perSubCondClust_onlyAelecs = cell(3,3,2);
Trials_perSubCondClust_onlyVelecs = cell(3,3,2);
Trials_perSubCondClust_A_AVelecs = cell(3,3,2);
Trials_perSubCondClust_inAllCond_elecs = cell(3,3,2);

inAllCond_clOne = cell(3,1);

for s = 1:3

    %take only the SMC nonnoisy channels
    ch = ismember(tvals(s).tVals(:,1),SMC_electrodes_nonnoisy{s});

    rest_length = min(restOnsets{s}(:,2) - restOnsets{s}(:,1));
    rest_traces = zeros(size(restOnsets{s},1),rest_length,sum(ch));
    for rt = 1:size(restOnsets{s},1)
        rest_traces(rt,:,:) = allData(s).ECoG_features(restOnsets{s}(rt,1):restOnsets{s}(rt,1)+(rest_length-1),ch);
    end

    % normalize the trials based on the preceding rest period
    trial_traces_Z = zeros(size(allData(s).trialTraces));
    for tr = 1:length(allData(s).trialLabels)

        % define which rest period start is closest to the VOT of each trial
        [~, restToPick] =  min(abs(restOnsets{s}(:,2) - allData(s).task_performance_data(allData(s).task_performance_data(:,4)==allData(s).trialLabels(tr,2),1)));

        mean_RestZ = mean(squeeze(rest_traces(restToPick,:,:)));
        std_RestZ = std(squeeze(rest_traces(restToPick,:,:)));

        trial_traces_Z(tr,:,:) = (squeeze(allData(s).trialTraces(tr,:,:)) - mean_RestZ)./ std_RestZ;

    end
    allData(s).trialTraces_Z = trial_traces_Z;

    % get the electrodes that are dorsally clustered in all conditions
    inAllCond_clOne{s} = intersect(elecs_per_cond_sub_Clust1{s,1}, elecs_per_cond_sub_Clust1{s,2});
    inAllCond_clOne{s} = intersect(inAllCond_clOne{s}, elecs_per_cond_sub_Clust1{s,3});

    for c = 1:3
        avgOverTime = squeeze(mean(trial_traces_Z,2));

        allElec_clOne = cluster_one_elecs{s,c};
        allElec_clOne= unique(allElec_clOne);

        allElec_clTwo = cluster_two_elecs{s,c};
        allElec_clTwo= unique(allElec_clTwo);

        AVElec_clOne =  elecs_per_cond_sub_Clust1{s,1};
        AVElec_clTwo =  elecs_per_cond_sub_Clust2{s,1};

        AElec_clOne =  elecs_per_cond_sub_Clust1{s,3};
        AElec_clTwo =  elecs_per_cond_sub_Clust2{s,3};

        VElec_clOne = elecs_per_cond_sub_Clust1{s,2};
        VElec_clTwo = elecs_per_cond_sub_Clust2{s,2};

        A_AVElec_clOne = [elecs_per_cond_sub_Clust1{s,1}; elecs_per_cond_sub_Clust1{s,3}];
        A_AVElec_clOne = unique(A_AVElec_clOne);

        A_AVElec_clTwo = [elecs_per_cond_sub_Clust2{s,1}; elecs_per_cond_sub_Clust2{s,3}];
        A_AVElec_clTwo = unique(A_AVElec_clTwo);

        % take all the trials in this condition, in the specified channels
        Trials_perSubCondClust{s,c,1} = mean(avgOverTime(allData(s).trialLabels(:,1)==c,ismember(SMC_electrodes_nonnoisy{s},allElec_clOne)),2);
        Trials_perSubCondClust{s,c,2} = mean(avgOverTime(allData(s).trialLabels(:,1)==c,ismember(SMC_electrodes_nonnoisy{s}, allElec_clTwo)),2);

        Trials_perSubCondClust_notAvg{s,c,1} = avgOverTime(allData(s).trialLabels(:,1)==c,ismember(SMC_electrodes_nonnoisy{s},allElec_clOne));
        Trials_perSubCondClust_notAvg{s,c,2} = avgOverTime(allData(s).trialLabels(:,1)==c,ismember(SMC_electrodes_nonnoisy{s}, allElec_clTwo));

        Trials_perSubCondClust_onlyAVelecs{s,c,1} = mean(avgOverTime(allData(s).trialLabels(:,1)==c, ismember(SMC_electrodes_nonnoisy{s}, AVElec_clOne)),2);
        Trials_perSubCondClust_onlyAVelecs{s,c,2} = mean(avgOverTime(allData(s).trialLabels(:,1)==c, ismember(SMC_electrodes_nonnoisy{s}, AVElec_clTwo)),2);

        Trials_perSubCondClust_onlyAelecs{s,c,1} = mean(avgOverTime(allData(s).trialLabels(:,1)==c, ismember(SMC_electrodes_nonnoisy{s}, AElec_clOne)),2);
        Trials_perSubCondClust_onlyAelecs{s,c,2} = mean(avgOverTime(allData(s).trialLabels(:,1)==c, ismember(SMC_electrodes_nonnoisy{s}, AElec_clTwo)),2);

        Trials_perSubCondClust_onlyVelecs{s,c,1} = mean(avgOverTime(allData(s).trialLabels(:,1)==c, ismember(SMC_electrodes_nonnoisy{s}, VElec_clOne)),2);
        Trials_perSubCondClust_onlyVelecs{s,c,2} = mean(avgOverTime(allData(s).trialLabels(:,1)==c, ismember(SMC_electrodes_nonnoisy{s}, VElec_clTwo)),2);

        Trials_perSubCondClust_A_AVelecs{s,c,1} = mean(avgOverTime(allData(s).trialLabels(:,1)==c, ismember(SMC_electrodes_nonnoisy{s}, A_AVElec_clOne)),2);
        Trials_perSubCondClust_A_AVelecs{s,c,2} = mean(avgOverTime(allData(s).trialLabels(:,1)==c, ismember(SMC_electrodes_nonnoisy{s}, A_AVElec_clTwo)),2);

        Trials_perSubCondClust_inAllCond_elecs{s,c,1} = mean(avgOverTime(allData(s).trialLabels(:,1) ==c , ismember(SMC_electrodes_nonnoisy{s}, inAllCond_clOne{s})),2);
    end
end

%% Figure 5: Plot the HFB traces within the dorsal cluster and do significance testing for WITHIN cluster differences in HFB power between the conditions

wndow = round(-0.5*500):500; % take a window of 0.5 second before and 1 second after Doremi onset time
class_window = 0:round(0.7*500); % period after VOT: 700 ms
period_to_indicate = find(wndow == 0, 1, 'first') + class_window;
lineStyle{1} = 'k-';  lineStyle{2} = 'k:';  lineStyle{3} = 'k--';  

%-------- Cluster one: Dorsal
clustOneAll = [];
clustOneGroups = [];
for c = 1:3
    trls = [Trials_perSubCondClust{1,c,1}; Trials_perSubCondClust{2,c,1}; Trials_perSubCondClust{3,c,1}];
    clustOneAll = [clustOneAll; trls]; %#ok<AGROW>
    clustOneGroups = [clustOneGroups; ones(length(trls),1)*c]; %#ok<AGROW>
end

figure; boxplot(clustOneAll, clustOneGroups);
ylim([-1.2 1.6]);
[~, ~, stats] = anova1(clustOneAll, clustOneGroups);
multcompare(stats)

% Plot the response over time in these electrodes
tot_el_clusterOne = length(elecsToTake_all_Clust1);

Clust1_all_traces = zeros(3,size(cluster_one_traces{1,1},2));
for c = 1:3
    perSub = zeros(3,size(cluster_one_traces{1,1},2));
    for s = 1:3
        all_Elecs = [elecs_per_cond_sub_Clust1{s,1}; elecs_per_cond_sub_Clust1{s,2}; elecs_per_cond_sub_Clust1{s,3}];
        all_Elecs = unique(all_Elecs);
        perSub(s,:) = squeeze(mean(allData(s).avgTrialTraces(c,:,ismember(SMC_electrodes_nonnoisy{s},all_Elecs)) .* (length(all_Elecs)/ tot_el_clusterOne),3));
    end
    Clust1_all_traces(c,:) = mean(perSub);
end

figure;
ylim([-0.1 0.3])
ylims = ylim;
patch([period_to_indicate fliplr(period_to_indicate)], [ones(1,length(period_to_indicate))*ylims(1) fliplr(ones(1,length(period_to_indicate))*ylims(2))], [0.7 0.7 0.7], 'LineStyle', 'none', 'EdgeColor', [0.7 0.7 0.7])
hold on;
title('Dorsal AV, V and A Electrodes')
xline(find(wndow==0, 1, 'first'), '--k')
xticks(1:round(0.25*500):length(wndow))
xticklabels(round(wndow(1)/500,2):0.25:round(wndow(end)/500,2));
for c= 1:3
   plot(Clust1_all_traces(c,:), lineStyle{c}, 'LineWidth', 3) ;
end


%% Sanity check: Do for the dorsal cluster all groups seperately

clustOneAll_AV = [];
clustOneGroups_AV = [];
for c = 1:3
    trls = [Trials_perSubCondClust_onlyAVelecs{1,c,1}; Trials_perSubCondClust_onlyAVelecs{2,c,1}; Trials_perSubCondClust_onlyAVelecs{3,c,1}];
    clustOneAll_AV = [clustOneAll_AV; trls]; %#ok<AGROW>
    clustOneGroups_AV = [clustOneGroups_AV; ones(length(trls),1)*c]; %#ok<AGROW>
end

figure; boxplot(clustOneAll_AV, clustOneGroups_AV); ylim([-1.2 1.6]);
[~, ~, stats] = anova1(clustOneAll_AV, clustOneGroups_AV);
multcompare(stats)

% plot the traces over time
totElecs = length(elecs_per_cond_sub_Clust1{1,1}) + length(elecs_per_cond_sub_Clust1{2,1})  + length(elecs_per_cond_sub_Clust1{3,1}) ;

Clust1_AV_traces = zeros(3,size(cluster_one_traces{1,1},2));
for c = 1:3

    perSub = zeros(3,size(cluster_two_traces{1,1},2));
    for s = 1:3
        AV_Elecs = elecs_per_cond_sub_Clust1{s,1};
        perSub(s,:) = squeeze(mean(allData(s).avgTrialTraces(c,:,ismember(SMC_electrodes_nonnoisy{s},AV_Elecs)) .* (length(AV_Elecs)/ totElecs),3));
    end
    Clust1_AV_traces(c,:) = mean(perSub);
end

figure;
ylim([-0.1 0.3])
ylims = ylim;
patch([period_to_indicate fliplr(period_to_indicate)], [ones(1,length(period_to_indicate))*ylims(1) fliplr(ones(1,length(period_to_indicate))*ylims(2))], [0.7 0.7 0.7], 'LineStyle', 'none', 'EdgeColor', [0.7 0.7 0.7])
hold on;
xline(find(wndow==0, 1, 'first'), '--k')
xticks(1:round(0.25*500):length(wndow))
xticklabels(round(wndow(1)/500,2):0.25:round(wndow(end)/500,2));
title('Dorsal AudioVisual Electrodes')
for c= 1:3
    plot(Clust1_AV_traces(c,:),'Color', plotcolors(c,:), 'LineWidth', 3) ;
end

%%%%%%%% VIDEO ONLY

clustOneAll_V = [];
clustOneGroups_V = [];
for c = 1:3
    trls = [Trials_perSubCondClust_onlyVelecs{1,c,1}; Trials_perSubCondClust_onlyVelecs{2,c,1}; Trials_perSubCondClust_onlyVelecs{3,c,1}];
    clustOneAll_V = [clustOneAll_V; trls]; %#ok<AGROW>
    clustOneGroups_V = [clustOneGroups_V; ones(length(trls),1)*c]; %#ok<AGROW>
end

figure; boxplot(clustOneAll_V, clustOneGroups_V); ylim([-1.2 1.6]);
[~, ~, stats] = anova1(clustOneAll_V, clustOneGroups_V);
multcompare(stats)

% plot the traces over time
totElecs = length(elecs_per_cond_sub_Clust1{1,2}) + length(elecs_per_cond_sub_Clust1{2,2})  + length(elecs_per_cond_sub_Clust1{3,2}) ;

Clust1_V_traces = zeros(3,size(cluster_one_traces{1,1},2));
for c = 1:3

    perSub = zeros(3,size(cluster_two_traces{1,1},2));
    for s = 1:3
        V_Elecs = elecs_per_cond_sub_Clust1{s,2};
        perSub(s,:) = squeeze(mean(allData(s).avgTrialTraces(c,:,ismember(SMC_electrodes_nonnoisy{s},V_Elecs)) .* (length(V_Elecs)/ totElecs),3));
    end
    Clust1_V_traces(c,:) = mean(perSub);
end

figure;
ylim([-0.1 0.3])
ylims = ylim;
patch([period_to_indicate fliplr(period_to_indicate)], [ones(1,length(period_to_indicate))*ylims(1) fliplr(ones(1,length(period_to_indicate))*ylims(2))], [0.7 0.7 0.7], 'LineStyle', 'none', 'EdgeColor', [0.7 0.7 0.7])
hold on;
xline(find(wndow==0, 1, 'first'), '--k')
xticks(1:round(0.25*500):length(wndow))
xticklabels(round(wndow(1)/500,2):0.25:round(wndow(end)/500,2));
title('Dorsal Visual Electrodes')
for c= 1:3
    plot(Clust1_V_traces(c,:),'Color', plotcolors(c,:), 'LineWidth', 3) ;
end

%%%%%%% AUDIO ONLY

clustOneAll_A = [];
clustOneGroups_A = [];
for c = 1:3
    trls = [Trials_perSubCondClust_onlyAelecs{1,c,1}; Trials_perSubCondClust_onlyAelecs{2,c,1}; Trials_perSubCondClust_onlyAelecs{3,c,1}];
    clustOneAll_A = [clustOneAll_A; trls]; %#ok<AGROW>
    clustOneGroups_A = [clustOneGroups_A; ones(length(trls),1)*c]; %#ok<AGROW>
end

figure; boxplot(clustOneAll_A, clustOneGroups_A); ylim([-1.2 1.6]);
[~, ~, stats] = anova1(clustOneAll_A, clustOneGroups_A);
multcompare(stats)

% plot the traces over time
totElecs = length(elecs_per_cond_sub_Clust1{1,3}) + length(elecs_per_cond_sub_Clust1{2,3})  + length(elecs_per_cond_sub_Clust1{3,3}) ;

Clust1_A_traces = zeros(3,size(cluster_one_traces{1,1},2));
for c = 1:3

    perSub = zeros(3,size(cluster_two_traces{1,1},2));
    for s = 1:3
        A_Elecs = elecs_per_cond_sub_Clust1{s,3};
        perSub(s,:) = squeeze(mean(allData(s).avgTrialTraces(c,:,ismember(SMC_electrodes_nonnoisy{s},A_Elecs)) .* (length(A_Elecs)/ totElecs),3));
    end
    Clust1_A_traces(c,:) = mean(perSub);
end

figure;
ylim([-0.1 0.3])
ylims = ylim;
patch([period_to_indicate fliplr(period_to_indicate)], [ones(1,length(period_to_indicate))*ylims(1) fliplr(ones(1,length(period_to_indicate))*ylims(2))], [0.7 0.7 0.7], 'LineStyle', 'none', 'EdgeColor', [0.7 0.7 0.7])
hold on;
xline(find(wndow==0, 1, 'first'), '--k')
xticks(1:round(0.25*500):length(wndow))
xticklabels(round(wndow(1)/500,2):0.25:round(wndow(end)/500,2));
title('Dorsal Audio Electrodes')
for c= 1:3
    plot(Clust1_A_traces(c,:),'Color', plotcolors(c,:), 'LineWidth', 3) ;
end


%% Figure 5: Plot the HFB traces within the ventral cluster and do significance testing for WITHIN cluster differences in HFB power between the conditions
lineStyle{1} = 'k-';  lineStyle{2} = 'k:';  lineStyle{3} = 'k--';  

clustTwoAll_A_AV = [];
clustTwoGroups_A_AV = [];
for c = 1:3
    trls = [Trials_perSubCondClust_A_AVelecs{1,c,2}; Trials_perSubCondClust_A_AVelecs{2,c,2}; Trials_perSubCondClust_A_AVelecs{3,c,2}];
    clustTwoAll_A_AV = [clustTwoAll_A_AV; trls]; %#ok<AGROW>
    clustTwoGroups_A_AV = [clustTwoGroups_A_AV; ones(length(trls),1)*c]; %#ok<AGROW>
end
figure; boxplot(clustTwoAll_A_AV, clustTwoGroups_A_AV); ylim([-1.2 1.6])

[~, ~, stats] = anova1(clustTwoAll_A_AV, clustTwoGroups_A_AV);
multcompare(stats)

totElecs = length(unique([elecs_per_cond_sub_Clust2{1,1}; elecs_per_cond_sub_Clust2{1,3}])) + length(unique([elecs_per_cond_sub_Clust2{2,1}; elecs_per_cond_sub_Clust2{2,3}])) + length(unique([elecs_per_cond_sub_Clust2{3,1}; elecs_per_cond_sub_Clust2{3,3}]));

Clust2_A_AV_traces = zeros(3,size(cluster_two_traces{1,1},2));
for c = 1:3

    perSub = zeros(3,size(cluster_two_traces{1,1},2));
    for s = 1:3
        A_AV_Elecs = [elecs_per_cond_sub_Clust2{s,1}; elecs_per_cond_sub_Clust2{s,3}];
        A_AV_Elecs = unique(A_AV_Elecs);
        perSub(s,:) = squeeze(mean(allData(s).avgTrialTraces(c,:,ismember(SMC_electrodes_nonnoisy{s},A_AV_Elecs)) .* (length(A_AV_Elecs)/ totElecs),3));
    end
    Clust2_A_AV_traces(c,:) = mean(perSub);
end

    figure;
    ylim([-0.1 0.3])
    ylims = ylim;
    patch([period_to_indicate fliplr(period_to_indicate)], [ones(1,length(period_to_indicate))*ylims(1) fliplr(ones(1,length(period_to_indicate))*ylims(2))], [0.7 0.7 0.7], 'LineStyle', 'none', 'EdgeColor', [0.7 0.7 0.7])
    hold on;
    title('Ventral A and AV Electrodes')
    xline(find(wndow==0, 1, 'first'), '--k')
    xticks(1:round(0.25*500):length(wndow))
    xticklabels(round(wndow(1)/500,2):0.25:round(wndow(end)/500,2));
    for c= 1:3
        plot(Clust2_A_AV_traces(c,:), lineStyle{c}, 'LineWidth', 3) ;
    end

%% Sanity check: Repeat statistical testing for the ventral cluster, but now only with AV and A electrodes separately

clustTwoAll_AV = [];
clustTwoGroups_AV = [];
for c = 1:3
    trls = [Trials_perSubCondClust_onlyAVelecs{1,c,2}; Trials_perSubCondClust_onlyAVelecs{2,c,2}; Trials_perSubCondClust_onlyAVelecs{3,c,2}];
    clustTwoAll_AV = [clustTwoAll_AV; trls]; %#ok<AGROW>
    clustTwoGroups_AV = [clustTwoGroups_AV; ones(length(trls),1)*c]; %#ok<AGROW>
end

figure; boxplot(clustTwoAll_AV, clustTwoGroups_AV); ylim([-1.2 1.6]);
[~, ~, stats] = anova1(clustTwoAll_AV, clustTwoGroups_AV);
multcompare(stats)

% plot the traces over time
totElecs = length(elecs_per_cond_sub_Clust2{1,1}) + length(elecs_per_cond_sub_Clust2{2,1})  + length(elecs_per_cond_sub_Clust2{3,1}) ;

Clust2_AV_traces = zeros(3,size(cluster_two_traces{1,1},2));
for c = 1:3

    perSub = zeros(3,size(cluster_two_traces{1,1},2));
    for s = 1:3
        AV_Elecs = elecs_per_cond_sub_Clust2{s,1};
        perSub(s,:) = squeeze(mean(allData(s).avgTrialTraces(c,:,ismember(SMC_electrodes_nonnoisy{s},AV_Elecs)) .* (length(AV_Elecs)/ totElecs),3));
    end
    Clust2_AV_traces(c,:) = mean(perSub);
end

figure;
ylim([-0.1 0.3])
ylims = ylim;
patch([period_to_indicate fliplr(period_to_indicate)], [ones(1,length(period_to_indicate))*ylims(1) fliplr(ones(1,length(period_to_indicate))*ylims(2))], [0.7 0.7 0.7], 'LineStyle', 'none', 'EdgeColor', [0.7 0.7 0.7])
hold on;
xline(find(wndow==0, 1, 'first'), '--k')
xticks(1:round(0.25*500):length(wndow))
xticklabels(round(wndow(1)/500,2):0.25:round(wndow(end)/500,2));
title('Ventral AudioVisual Electrodes')
for c= 1:3
    plot(Clust2_AV_traces(c,:),'Color', plotcolors(c,:), 'LineWidth', 3) ;
end

%----------------- do for the v clustered electrodes
clustTwoAll_V = [];
clustTwoGroups_V = [];
for c = 1:3
    trls = [Trials_perSubCondClust_onlyVelecs{1,c,2}; Trials_perSubCondClust_onlyVelecs{2,c,2}; Trials_perSubCondClust_onlyVelecs{3,c,2}];
    clustTwoAll_V = [clustTwoAll_V; trls]; %#ok<AGROW>
    clustTwoGroups_V = [clustTwoGroups_V; ones(length(trls),1)*c]; %#ok<AGROW>
end

figure; boxplot(clustTwoAll_V, clustTwoGroups_V); ylim([-1.2 1.6]);
[~, ~, stats] = anova1(clustTwoAll_V, clustTwoGroups_AV);
multcompare(stats)

% plot the traces over time
totElecs = length(elecs_per_cond_sub_Clust2{1,2}) + length(elecs_per_cond_sub_Clust2{2,2})  + length(elecs_per_cond_sub_Clust2{3,2}) ;

Clust2_V_traces = zeros(3,size(cluster_two_traces{1,1},2));
for c = 1:3

    perSub = zeros(3,size(cluster_two_traces{1,1},2));
    for s = 1:3
        V_Elecs = elecs_per_cond_sub_Clust2{s,2};
        perSub(s,:) = squeeze(mean(allData(s).avgTrialTraces(c,:,ismember(SMC_electrodes_nonnoisy{s},V_Elecs)) .* (length(V_Elecs)/ totElecs),3));
    end
    Clust2_V_traces(c,:) = mean(perSub);
end

figure;
ylim([-0.1 0.3])
ylims = ylim;
patch([period_to_indicate fliplr(period_to_indicate)], [ones(1,length(period_to_indicate))*ylims(1) fliplr(ones(1,length(period_to_indicate))*ylims(2))], [0.7 0.7 0.7], 'LineStyle', 'none', 'EdgeColor', [0.7 0.7 0.7])
hold on;
xline(find(wndow==0, 1, 'first'), '--k')
xticks(1:round(0.25*500):length(wndow))
xticklabels(round(wndow(1)/500,2):0.25:round(wndow(end)/500,2));
title('Ventral Visual Electrodes')
for c= 1:3
    plot(Clust2_V_traces(c,:),'Color', plotcolors(c,:), 'LineWidth', 3) ;
end

%---------------- do for the A clustered electrodes
clustTwoAll_A = [];
clustTwoGroups_A = [];
for c = 1:3
    trls = [Trials_perSubCondClust_onlyAelecs{1,c,2}; Trials_perSubCondClust_onlyAelecs{2,c,2}; Trials_perSubCondClust_onlyAelecs{3,c,2}];
    clustTwoAll_A = [clustTwoAll_A; trls]; %#ok<AGROW>
    clustTwoGroups_A = [clustTwoGroups_A; ones(length(trls),1)*c]; %#ok<AGROW>
end
figure; boxplot(clustTwoAll_A, clustTwoGroups_A); ylim([-1.2 1.6])

[~, ~, stats] = anova1(clustTwoAll_A, clustTwoGroups_A);
multcompare(stats)

totElecs = length(elecs_per_cond_sub_Clust2{1,3}) + length(elecs_per_cond_sub_Clust2{2,3})  + length(elecs_per_cond_sub_Clust2{3,3}) ;

Clust2_A_traces = zeros(3,size(cluster_two_traces{1,1},2));
for c = 1:3

    perSub = zeros(3,size(cluster_two_traces{1,1},2));
    for s = 1:3
        A_Elecs = elecs_per_cond_sub_Clust2{s,3};
        perSub(s,:) = squeeze(mean(allData(s).avgTrialTraces(c,:,ismember(SMC_electrodes_nonnoisy{s},A_Elecs)) .* (length(A_Elecs)/ totElecs),3));
    end
    Clust2_A_traces(c,:) = mean(perSub);
end

figure;
ylim([-0.1 0.3])
ylims = ylim;
patch([period_to_indicate fliplr(period_to_indicate)], [ones(1,length(period_to_indicate))*ylims(1) fliplr(ones(1,length(period_to_indicate))*ylims(2))], [0.7 0.7 0.7], 'LineStyle', 'none', 'EdgeColor', [0.7 0.7 0.7])
hold on;
title('Ventral Audio Electrodes')
xline(find(wndow==0, 1, 'first'), '--k')
xticks(1:round(0.25*500):length(wndow))
xticklabels(round(wndow(1)/500,2):0.25:round(wndow(end)/500,2));
for c= 1:3
    plot(Clust2_A_traces(c,:),'Color', plotcolors(c,:), 'LineWidth', 3) ;
end

