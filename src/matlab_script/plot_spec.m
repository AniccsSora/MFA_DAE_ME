ind = 1;
FeaDim = 257;
FrameSize = 512;
FrameRate =256
FFT_SIZE = 512;
spectrumFlag = 2;
file_num = '3_1';
method = 'D:\git-repo\qpzm_mfa_dae\src\senpai_data\heart_lung_sam2\mix';


%     file_name = '\source0\6dB\'
file_name = '\training_clean_心跳\0dB\'
a = sprintf('%s%s%s%s',method, file_name, file_num,'.wav')
audioData = audioread(a);
sr = 8000
FontSize=50;LineWidth=1;
h0 = plot(0:1/sr:(length(audioData)-1)/sr,(audioData)/std(audioData));
set(gca, 'YLim',[-10,10]);

% fig=plot(audioData)
% legend('boxoff');box off;
set(gca,'fontsize',FontSize,'Fontname','Times New Roman');
set(h0,'LineWidth',LineWidth);

% xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Ampulitude','fontsize',FontSize,'Fontname','Times New Roman');
a=sprintf('%s%swav%s',method,file_name,file_num,'.png');
saveas(h0,a)

shrink=0;
shift=0;

[powspectrum]  = Spectrum(audioData, FrameSize, FrameRate, FFT_SIZE, spectrumFlag);
spectrogram = log10(eps + powspectrum);
fig = imagesc(spectrogram(1:70,:));% axis xy;  colormap('jet');xlabel('Frame index');
colormap Jet;axis xy;box off;

% set(gca,'fontsize',FontSize,'Fontname','Times New Roman');
% xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
spectrogram = log10(eps + powspectrum);
fig=imagesc(spectrogram(1:70,:)); axis xy; colormap('jet');%xlabel('Frame index');
set(gca, 'YTick',[0,32,70]);
set(gca,'YTickLabel',{'0','500','1050'})
% set(gca,'ytick',[])
set(gca, 'XLim',[0,300]);

set(gca,'fontsize',40,'Fontname','Times New Roman');
a=sprintf('%s%s%s',method,file_name,file_num,'.png')
saveas(fig,a)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
file_name = '\training_clean_心跳\6dB\'
a=sprintf('%s%s%s%s',method, file_name,file_num,'.wav')
audioData = audioread(a);

FontSize=50;LineWidth=1;
h2 = plot(0:1/sr:(length(audioData)-1)/sr,audioData/std(audioData));
set(gca, 'YLim',[-10,10]);
% fig=plot(audioData)
% legend('boxoff');box off;
set(gca,'fontsize',FontSize,'Fontname','Times New Roman');
set(h2,'LineWidth',LineWidth);
% xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Ampulitude','fontsize',FontSize,'Fontname','Times New Roman');
a=sprintf('%s%swav%s',method,file_name,file_num,'.png')
saveas(h2,a)

[powspectrum]  = Spectrum(audioData, FrameSize, FrameRate, FFT_SIZE, spectrumFlag);
spectrogram = log10(eps + powspectrum);
fig=imagesc(spectrogram(1:70,:)); axis xy; colormap('jet');%xlabel('Frame index');
set(gca, 'YTick',[0,32,70]);
set(gca,'YTickLabel',{'0','500','1050'})
% set(gca,'ytick',[])
set(gca, 'XLim',[0,300]);

set(gca,'fontsize',40,'Fontname','Times New Roman');
a=sprintf('%s%s%s',method,file_name,file_num,'.png')
saveas(fig,a)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_name = '\training_clean_心跳\6dB\'
a=sprintf('%s%s%s%s',method, file_name,file_num,'.wav')
audioData = audioread(a);

FontSize=50;LineWidth=1;
h1 = plot(0:1/sr:(length(audioData)-1)/sr,audioData/std(audioData));
set(gca, 'YLim',[-10,10]);
% fig=plot(audioData)
% legend('boxoff');box off;
set(gca,'fontsize',FontSize,'Fontname','Times New Roman');
set(h1,'LineWidth',LineWidth);
% xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Ampulitude','fontsize',FontSize,'Fontname','Times New Roman');
a=sprintf('%s%swav%s',method,file_name,file_num,'.png')
saveas(h1,a)

[powspectrum]  = Spectrum(audioData, FrameSize, FrameRate, FFT_SIZE, spectrumFlag);
spectrogram = log10(eps + powspectrum);
fig=imagesc(spectrogram(1:70,:)); axis xy; colormap('jet');%xlabel('Frame index'); 

set(gca, 'YTick',[0,32,70]);
set(gca,'YTickLabel',{'0','500','1050'})
% set(gca,'ytick',[])
set(gca, 'XLim',[0,300]);

set(gca,'fontsize',40,'Fontname','Times New Roman');

a=sprintf('%s%s%s',method,file_name,file_num,'.png')
saveas(fig,a)

