% 生出時頻圖(spectrogram)
% test in vscode

% Parameters
audioRootDir = "D:\Git\qpzm_mfa_dae\src\dataset\audio_Healthy"
audioPath = fullfile(audioRootDir, "121_1b1_Tc_sc_Meditron.wav")

[y, sr] = audioread(audioPath);

[s, f, t] = stft(y, sr,...
'Window', hamming(2048,'symmetric'),...
'OverlapLength', 128,...
'OutputTimeDimension', 'downrows',...
'FrequencyRange', 'onesided');


