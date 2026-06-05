%% EEG-VR Dataset Processing Pipeline
% This script reads raw DSI-7 EEG CSV files, performs basic quality control,
% applies a 0.5-40 Hz band-pass filter, segments each recording into 1-s epochs,
% and computes FFT-based theta-band summaries (4-8 Hz).
%
% Before running, set rawRoot to your RAW EEGS folder.
%
% Example:
% rawRoot = 'D:\EEG_VR_Prof.Adil\Saif\RAW EEGS';
% outRoot = fullfile(pwd, 'results', 'matlab_full_run');
% run('matlab/run_eeg_vr_pipeline.m');

if ~exist('rawRoot','var')
    rawRoot = 'D:\EEG_VR_Prof.Adil\Saif\RAW EEGS';
end
if ~exist('outRoot','var')
    outRoot = fullfile(pwd, 'results', 'matlab_full_run');
end
if ~exist(outRoot,'dir'); mkdir(outRoot); end
tableDir = fullfile(outRoot, 'tables');
figDir = fullfile(outRoot, 'figures');
if ~exist(tableDir,'dir'); mkdir(tableDir); end
if ~exist(figDir,'dir'); mkdir(figDir); end

channels = {'PzLE','F4LE','C4LE','P4LE','P3LE','C3LE','F3LE'};
fallbackFs = 300;

files = dir(fullfile(rawRoot, '**', '*.csv'));
if isempty(files)
    error('No CSV files found under: %s', rawRoot);
end

qcRows = {};
statRows = {};
thetaRows = {};
figureSaved = false;

for i = 1:numel(files)
    filePath = fullfile(files(i).folder, files(i).name);
    fprintf('Processing %s\n', filePath);

    lowerPath = lower(filePath);
    group = 'unknown';
    condition = 'unknown';
    if contains(lowerPath, 'control'); group = 'control'; end
    if contains(lowerPath, 'triathlete') || contains(lowerPath, 'athlete'); group = 'triathlete'; end
    if contains(lowerPath, 'baseline'); condition = 'baseline'; end
    if contains(lowerPath, [filesep 'vr']) || endsWith(lowerPath, [filesep 'vr' filesep files(i).name]); condition = 'vr'; end

    T = readtable(filePath, 'VariableNamingRule', 'preserve');
    varNames = T.Properties.VariableNames;
    present = channels(ismember(channels, varNames));
    if isempty(present)
        warning('No expected EEG channels in %s. Skipping.', files(i).name);
        continue;
    end

    X = table2array(T(:, present));
    X = double(X);
    if any(isnan(X(:)))
        for c = 1:size(X,2)
            col = X(:,c);
            col(isnan(col)) = median(col, 'omitnan');
            X(:,c) = col;
        end
    end

    [fs, fsSource] = inferFsFromTable(T, fallbackFs);
    nSamples = size(X,1);
    durationSec = nSamples / fs;

    triggerValues = 'not_available';
    if ismember('Trigger', varNames)
        tr = T.Trigger;
        try
            u = unique(tr);
            if numel(u) > 20; u = u(1:20); end
            triggerValues = strjoin(string(u), ';');
        catch
            triggerValues = 'present_unreadable';
        end
    end

    qcRows(end+1,:) = {files(i).name, group, condition, nSamples, numel(present), strjoin(present,';'), fs, fsSource, durationSec, durationSec/60, sum(isnan(X(:))), triggerValues}; %#ok<SAGROW>

    for c = 1:numel(present)
        col = X(:,c);
        statRows(end+1,:) = {files(i).name, group, condition, present{c}, mean(col,'omitnan'), std(col,'omitnan'), min(col), max(col)}; %#ok<SAGROW>
    end

    X = X - mean(X,1,'omitnan');
    XF = bandpassFilterEEG(X, fs, 0.5, 40);
    thetaTable = computeThetaSummary(XF, fs, present, files(i).name, group, condition);
    thetaRows = [thetaRows; thetaTable]; %#ok<AGROW>

    if ~figureSaved
        createRepresentativeFigures(XF, fs, present, figDir);
        figureSaved = true;
    end
end

qcTable = cell2table(qcRows, 'VariableNames', {'filename','group','condition','n_samples','n_channels_present','channels_present','estimated_fs_hz','fs_source','duration_seconds','duration_minutes','missing_eeg_values','trigger_values'});
writetable(qcTable, fullfile(tableDir, 'quality_control_summary.csv'));

statTable = cell2table(statRows, 'VariableNames', {'filename','group','condition','channel','mean_raw','sd_raw','min_raw','max_raw'});
writetable(statTable, fullfile(tableDir, 'channel_raw_statistics.csv'));

if ~isempty(thetaRows)
    thetaTableAll = cell2table(thetaRows, 'VariableNames', {'filename','group','condition','channel','n_epochs_1s','theta_abs_mean','theta_abs_sd','theta_relative_mean','theta_relative_sd'});
    writetable(thetaTableAll, fullfile(tableDir, 'theta_power_summary.csv'));
end

fprintf('Done. Results saved in: %s\n', outRoot);

%% Local helper functions
function [fs, source] = inferFsFromTable(T, fallbackFs)
    fs = fallbackFs;
    source = 'fallback';
    vars = T.Properties.VariableNames;
    if ismember('DeviceTimeStamp', vars)
        t = T.DeviceTimeStamp;
        t = double(t(~isnan(t)));
        if numel(t) > 10
            dt = diff(t(1:min(numel(t),5000)));
            dt = dt(dt > 0 & isfinite(dt));
            if ~isempty(dt)
                candidate = 1 / median(dt);
                if candidate >= 50 && candidate <= 2000
                    fs = round(candidate, 3);
                    source = 'DeviceTimeStamp';
                    return;
                end
            end
        end
    end
    if ismember('DeviceTimeUnixTimeStamp', vars)
        t = T.DeviceTimeUnixTimeStamp;
        t = double(t(~isnan(t)));
        if numel(t) > 10
            dt = diff(t(1:min(numel(t),5000))) / 1000;
            dt = dt(dt > 0 & isfinite(dt));
            if ~isempty(dt)
                candidate = 1 / median(dt);
                if candidate >= 50 && candidate <= 2000
                    fs = round(candidate, 3);
                    source = 'DeviceTimeUnixTimeStamp';
                end
            end
        end
    end
end

function XF = bandpassFilterEEG(X, fs, lowHz, highHz)
    % Requires Signal Processing Toolbox for butter/filtfilt.
    [b,a] = butter(4, [lowHz highHz] / (fs/2), 'bandpass');
    XF = filtfilt(b, a, X);
end

function thetaRows = computeThetaSummary(XF, fs, channels, filename, group, condition)
    nPer = round(fs);
    nEpochs = floor(size(XF,1) / nPer);
    thetaRows = {};
    if nEpochs < 1; return; end

    Xtrim = XF(1:nEpochs*nPer, :);
    freqs = (0:(nPer/2)) * (fs/nPer);
    thetaIdx = freqs >= 4 & freqs <= 8;
    totalIdx = freqs >= 0.5 & freqs <= 40;

    thetaPower = zeros(nEpochs, numel(channels));
    totalPower = zeros(nEpochs, numel(channels));

    for e = 1:nEpochs
        idx = (e-1)*nPer + (1:nPer);
        epoch = Xtrim(idx, :);
        epoch = epoch - mean(epoch, 1);
        Y = fft(epoch);
        P = abs(Y(1:numel(freqs), :)).^2 / nPer;
        thetaPower(e,:) = sum(P(thetaIdx,:),1);
        totalPower(e,:) = sum(P(totalIdx,:),1);
    end

    relTheta = thetaPower ./ max(totalPower, eps);
    for c = 1:numel(channels)
        thetaRows(end+1,:) = {filename, group, condition, channels{c}, nEpochs, mean(thetaPower(:,c)), std(thetaPower(:,c)), mean(relTheta(:,c)), std(relTheta(:,c))}; %#ok<AGROW>
    end
end

function createRepresentativeFigures(XF, fs, channels, figDir)
    sec = 5;
    n = min(round(sec*fs), size(XF,1));
    t = (0:n-1) / fs;
    offset = std(XF(1:n,:), 0, 'all') * 6;
    if offset == 0 || isnan(offset); offset = 100; end

    f = figure('Visible','off','Color','w','Position',[100 100 1100 600]);
    hold on;
    for c = 1:numel(channels)
        y = XF(1:n,c) + (numel(channels)-c)*offset;
        plot(t, y, 'k', 'LineWidth', 0.8);
        text(-0.08, (numel(channels)-c)*offset, erase(channels{c},'LE'), 'HorizontalAlignment','right');
    end
    xlabel('Time (s)');
    yticks([]); xlim([0 sec]);
    title('Representative filtered EEG segment (0.5-40 Hz)');
    exportgraphics(f, fullfile(figDir, 'representative_eeg_segment.png'), 'Resolution', 300);
    close(f);

    % PSD using pwelch on each channel
    f2 = figure('Visible','off','Color','w','Position',[100 100 900 550]);
    hold on;
    psdAll = [];
    for c = 1:numel(channels)
        [pxx, freq] = pwelch(XF(:,c), round(2*fs), [], [], fs);
        psdAll(:,c) = pxx; %#ok<AGROW>
    end
    meanPsd = mean(psdAll,2);
    semilogy(freq, meanPsd, 'LineWidth', 1.8);
    xline(4, '--'); xline(8, '--');
    xlim([0 40]); grid on;
    xlabel('Frequency (Hz)'); ylabel('Power (a.u./Hz)');
    title('Representative power spectral density with theta band markers');
    exportgraphics(f2, fullfile(figDir, 'representative_psd_theta_band.png'), 'Resolution', 300);
    close(f2);
end
