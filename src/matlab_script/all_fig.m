

close all;

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
 method = 'D:\import_data\heart_lung_sam2_exp_eval\1file_heart_lung_sam2_0222\HL_FFTseparate';
%  method = 'D:\heart_lung_sam2_exp_eval\heart_lung_sam2_NMF_0215\HL_FFTseparate';
 method = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2_NMF_0318\HL_FFTseparate';
 
% % 
%   method = 'D:\heart_lung_sam2_exp_eval\heart_lung_sam2\mix';
%  method = 'D:\heart_lung_sam2_exp_eval\1file_sam2_DDAE\HL_FFTseparate';
% 
%   method = 'D:\heart_lung_sam2_exp_eval\1file_heart_lung_sam2_0319\HL_FFTseparate';
%  method= 'D:\heart_lung_sam2_exp_eval\1file_heart_lung_sam2_0319\HLseparate';
FontSize = 10
figure(1)
wavfilename1 ='D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source0\6dB\4_1.wav';
subplot(221);
[CleanSpec,sig] = wav2spec(wavfilename1);
h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
colormap Jet;axis xy;box off;
set(gca, 'CLim', [1 40]);%frequency band
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel('NMF','fontsize',FontSize,'Fontname','Times New Roman');
% % xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');


subplot(222)
wavfilename1 ='D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source1\6dB\4_1.wav';
[CleanSpec,sig] = wav2spec(wavfilename1);
h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
% h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
 
colormap Jet;axis xy;box off;
set(gca, 'CLim', [1 40]);%frequency band

set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
% set(gca, 'YTick',[0,20, 40]);
% set(gca,'YTickLabel',{'0','500','1050'})
wavfilename1 ='D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source0\6dB\4_1.wav';
subplot(223);
[CleanSpec,sig] = wav2spec(wavfilename1);
h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
colormap Jet;axis xy;box off;
set(gca, 'CLim', [1 40]);%frequency band
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');


subplot(224)
wavfilename1 ='D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source1\6dB\4_1.wav';
[CleanSpec,sig] = wav2spec(wavfilename1);
h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
% h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
 
colormap Jet;axis xy;box off;
set(gca, 'CLim', [1 40]);%frequency band
set(gca, 'YTick',[0, 5, 10]);
set(gca,'YTickLabel',{'0','500','1050'})
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');


figure(2)
wavfilename1 ='D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2_NMF_0318\HL_FFTseparate\source0\6dB\4_1.wav';
subplot(221);
[CleanSpec,sig] = wav2spec(wavfilename1);
h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
colormap Jet;axis xy;box off;
set(gca, 'CLim', [1 30]);%frequency band
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');


subplot(222)
wavfilename1 ='D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2_NMF_0318\HL_FFTseparate\source1\6dB\4_1.wav';
[CleanSpec,sig] = wav2spec(wavfilename1);
h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
% h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
imresize(h_pc,[20 40]);  
colormap Jet;axis xy;box off;
set(gca, 'CLim', [1 30]);%frequency band
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel('Time (s)','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');







