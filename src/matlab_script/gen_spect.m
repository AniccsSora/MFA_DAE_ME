% 讀檔

% "D:\Git\qpzm_mfa_dae\src\dataset\audio_Healthy";
audioRootDir = "D:\Git\qpzm_mfa_dae\src\senpai_data\heart_lung_sam2\mix\training_noisy\0dB";  

% "121_1b1_Tc_sc_Meditron.wav"
audioPath = fullfile(audioRootDir, "8_1.wav");  

[y, sr] = audioread(audioPath);

timetable_y = timetable(y,'SampleRate', sr);

% Compute spectrogram
% Parameters
timeLimits = seconds([0 9.999875]); % seconds
frequencyLimits = [0 4000]; % Hz
leakage = 0;
timeResolution = 0.05; % seconds
overlapPercent = 60;

% Index into signal time region of interest
timetable_y_y_ROI = timetable_y(:,'y');
timetable_y_y_ROI = timetable_y_y_ROI(timerange(timeLimits(1),timeLimits(2),'closed'),1);

% Compute spectral estimate
% Run the function call below without output arguments to plot the results
figure('Name', 'STFT spectrogram',...
'NumberTitle', 'off');
pspectrum(timetable_y_y_ROI, ...
    'spectrogram', ...
    'FrequencyLimits',frequencyLimits, ...
    'Leakage',leakage, ...
    'TimeResolution',timeResolution, ...
    'OverlapPercent',overlapPercent);

cmap=readmatrix('C:\Users\Anicca\Desktop\colormap\save_colormap.txt');
colormap(cmap)

