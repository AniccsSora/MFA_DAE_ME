function sig=PowerSpectrum2Wave(log10powerspectrum,yphase)
%sig=InversePowerSpectrum(log10powerspectrum,yphase)
%log10powerspectrum: estimated from deep autoencoder, must be 129*frames,
%must be log10 compressed
%yphase: clean or noisy phase information, must be a matrix as 256*frames
%Xugang Lu @NICT


logpowspectrum             =log(power(10,log10powerspectrum)); %log power spectrum
%yphase                     =yphase(1:128+1,:); %For Odd sample,  for 256 

% 計算  logpowspectrum rows columns 數。
[row, col]=size(logpowspectrum);

% yphase                     =yphase(1:128+1,:); %For Odd sample  寫死 專給 wav2spec.m 的  FrameLength=256; 時候所用。
%yphase_debug                     =yphase(1:128+1,:); %For Odd sample  寫死 專給 wav2spec.m 的  FrameLength=256; 時候所用。
yphase                     =yphase(1:row,:); %For Odd sample  動態desu

sig                        =OverlapAdd(sqrt(exp(logpowspectrum)-0.01),yphase);

return;
