function [CleanSpec,sig] = wav2spec(wavfilename)

% ��l�ƭ�
% FrameLength=256;
% FrameRate=40;
FrameLength=1024;
FrameRate=16; % shift
FFT_SIZE=FrameLength;
flag=1;% ��l��1
% sr=5000;
sr=88879879874698469; % �ش��S�ιL


shift=0;
shrink=0;


%%%%%%%%%%%%%%%%%%%%%%

% wav='.\noisy.wav'; % clean
% wav='C:\Users\frank.chang\Desktop\HeartSound_noisy\heart_sound_noise_5k_samelength.wav'; % clean
% wav='C:\Users\frank.chang\Desktop\HeartSound_noisy\paper\spectrum\heart_sound_noise_5k_samelength.wav'; % clean
% wav='D:\�Ф�ftp���\�x�j�`�|\IRB �������׾�z\HH0084\005#SA_HH0084_VSD0.wav';

x=audioread(wavfilename);
%  x=downsample(x,4)
x=x/std(x);
% x=audioread(wav);x=resample(x,1000,5000);x=x/std(x);

% [CleanSpec,~,yphase] = Spectrum(x*1000,FrameLength,FrameRate,FFT_SIZE, flag);
[CleanSpec,~,yphase] = Spectrum(x*500,FrameLength,FrameRate,FFT_SIZE, flag);

log10powerspectrum  =log10(CleanSpec);fs=sr;
sig=PowerSpectrum2Wave(log10powerspectrum,yphase);%,FrameLength,FrameRate);
% sig=sig/std(sig);










