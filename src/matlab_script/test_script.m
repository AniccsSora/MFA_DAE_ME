
% ./_4_0recons.wav
audiopath_mix='D:/Git/qpzm_mfa_dae/src/senpai_data/heart_lung_sam2/mix/training_noisy_心肺/0dB/4_0.wav';
source1='D:/Git/qpzm_mfa_dae/src/matlab_script/_4_0_source1.wav';
source2='D:/Git/qpzm_mfa_dae/src/matlab_script/_4_0_source2.wav';

% [y, fs]=audioread(audiopath);

% [spec, x_seg, yphase]=Spectrum(y, 512, 256, 512, 2);

% spectrogram=log10(eps+spec);

% fig = imagesc(spectrogram(1:70,:));
% colormap Jet;
% axis xy;
% box off;

tiledlayout(3,1) % Requires R2019b or later


% mix
nexttile
[y, ~]=audioread(audiopath_mix);
[spec, ~, ~]=Spectrum(y, 512, 256, 512, 2);
spectrogram=log10(eps+spec);
%fig_mix = imagesc(spectrogram(1:70,:));
imagesc(spectrogram(1:70,:));
%colormap Jet;
axis xy;
box off;
%mixCB=colorbar;
title('mix')

% s1
nexttile
[y, ~]=audioread(source1);
[spec, ~, ~]=Spectrum(y, 512, 256, 512, 2);
spectrogram=log10(eps+spec);
%fig_s1 = imagesc(spectrogram(1:70,:));
imagesc(spectrogram(1:70,:));
%colormap Jet;
axis xy;
box off;
%s1CB=colorbar(mixCB);
title('source 1')
%s1CB.Visible='off'
% 
% % s2
nexttile
[y, ~]=audioread(source2);
[spec, ~, ~]=Spectrum(y, 512, 256, 512, 2);
spectrogram=log10(eps+spec);
%fig_s2 = imagesc(spectrogram(1:70,:));
imagesc(spectrogram(1:70,:));
%colormap Jet;
axis xy;
box off;
%s2CB=colorbar;
%s2CB.Visible='off';
title('source 2')

% 
colormap Jet;
cb = colorbar;
cb.Layout.Tile = 'east';



