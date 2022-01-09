function TestInverseSpectrum_MelSpectrum
%1. Extracing Mel spectrum, and save the phase information
%2. Do inverse transform to get wavform
%Xugang Lu @NICT
%March, 2013

%1. Mel spectrum extraction from wav file
fs  =16000;
x=wavread('Test.wav');
[Log_MFCSpectrum,yphase]=Mel_Spectrum_FromX(x*1000,2,256,128,256);  
%

%2. Do inverse transform from Mel spectrum to get wave form
MelSpec             =power(10,Log_MFCSpectrum);
[spec,wts,iwts]     =MelSpectrum2PowerSpectrum(MelSpec, 16000, 256, 'htkmel', 120,8000, 1, 1);
log10powerspectrum  =log10(spec);

sig=PowerSpectrum2Wave(log10powerspectrum,yphase);
siga=sig/max(abs(sig)); 
wavwrite(siga,fs,'MelInverse.wav');
subplot(3,1,1);
imagesc(Log_MFCSpectrum);
subplot(3,1,2);
imagesc(log10powerspectrum);
subplot(3,1,3);
plot(siga);
soundsc(siga,fs);

return