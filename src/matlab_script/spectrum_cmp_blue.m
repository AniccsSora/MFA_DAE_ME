function spectrum_cmp_blue(wav, )

% clear all;

% FrameLength=256;
% FrameRate=40;
FrameLength=256;
FrameRate=40;
FFT_SIZE=FrameLength;
flag=1;
% sr=5000;
sr=8000;

shift=0;
shrink=0;


%%%%%%%%%%%%%%%%%%%%%%

% wav='.\noisy.wav'; % clean
% wav='C:\Users\frank.chang\Desktop\HeartSound_noisy\heart_sound_noise_5k_samelength.wav'; % clean
% wav='C:\Users\frank.chang\Desktop\HeartSound_noisy\paper\spectrum\heart_sound_noise_5k_samelength.wav'; % clean
% wav='D:\創心ftp資料\台大總院\IRB 正式收案整理\HH0084\005#SA_HH0084_VSD0.wav';
% wav='C:\Users\frank.chang\Desktop\新增資料夾\2k\005#SA_HH0084_VSD0.wav';
% wav ='D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source2\-2dB\4_1.wav';
x=audioread(wav);x=x/std(x);
% x=audioread(wav);x=resample(x,1000,5000);x=x/std(x);

% [CleanSpec,~,yphase] = Spectrum(x*1000,FrameLength,FrameRate,FFT_SIZE, flag);
[CleanSpec,~,yphase] = Spectrum(x*500,FrameLength,FrameRate,FFT_SIZE, flag);

log10powerspectrum  =log10(CleanSpec);fs=sr;
sig=PowerSpectrum2Wave(log10powerspectrum,yphase,FrameLength,FrameRate);
% sig=sig/std(sig);

figure(1);
FontSize=50;LineWidth=1;

subplot(211)
h=plot(0:1/sr:(length(sig)-1)/sr,sig/std(sig));
legend('Signal');legend('boxoff');box off;
set(gca,'fontsize',FontSize,'Fontname','Times New Roman');
set(h,'LineWidth',LineWidth);
xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Ampulitude','fontsize',FontSize,'Fontname','Times New Roman');
figure(2);
subplot(111)
h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
colormap Jet;axis xy;box off;
set(gca, 'CLim', [1 60]);
% set(gca,'fontsize',FontSize,'Fontname','Times New Roman');
% xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
% set(gca,'FontWeight','bold','fontsize',20);

end




