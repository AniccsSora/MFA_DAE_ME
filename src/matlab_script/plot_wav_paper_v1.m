clear;
FrameLength=256;
FrameRate=40;
FFT_SIZE=FrameLength;
flag=1;

sr=2000;

shift=0;
shrink=0;
wavfilename1 = 'D:\git-repo\MFA_DAE_ME\src\dataset\224_1b1_Tc_sc_Meditron.wav'
wavfilename2 = 'D:\git-repo\MFA_DAE_ME\src\dataset\224_1b1_Tc_sc_Meditron.wav'

% wavfilename1 = 'D:\import_data\heart_lung_sam2_exp_eval\1file_heart_lung_sam2_0319\HL_FFTseparate\source2\6dB\4_1.wav'
% wavfilename2 = 'D:\import_data\heart_lung_sam2_exp_eval\1file_heart_lung_sam2_0319\HL_FFTseparate\source0\6dB\4_1.wav'
wavfilename1 = 'D:\git-repo\MFA_DAE_ME\src\dataset\224_1b1_Tc_sc_Meditron.wav'
wavfilename2 = 'D:\git-repo\MFA_DAE_ME\src\dataset\224_1b1_Tc_sc_Meditron.wav'

% wavfilename1 = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2_NMF_0318\HL_FFTseparate\source0\6dB\4_1.wav'
% wavfilename2 = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2_NMF_0318\HL_FFTseparate\source1\6dB\4_1.wav'
% % 
% wavfilename1 = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2_NMF_0215\HL_FFTseparate\source0\6dB\4_1.wav'
% wavfilename2 = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2_NMF_0215\HL_FFTseparate\source1\6dB\4_1.wav'

% 
 wavfilename1 = 'D:\��\Heartlung\data\1file_heart_lung_sam2_2020-3-12_1_400_0.01_1024_1025_DCAE_relu_tanh_2_PC_NMF_cluster2\HL_FFTseparate\source1\6dB\4_1.wav'
 wavfilename2 = 'D:\��\Heartlung\data\1file_heart_lung_sam2_2020-3-12_1_400_0.01_1024_1025_DCAE_relu_tanh_2_PC_NMF_cluster2\HL_FFTseparate\source1\6dB\4_1.wav'
% % 
%   wavfilename1 = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source0\6dB\4_1.wav'
%  wavfilename2 = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source1\6dB\4_1.wav'
%
exp_path = 'D:\exp_fig\'
wavfilename1 = 'D:\git-repo\MFA_DAE_ME\src\dataset\224_1b1_Tc_sc_Meditron.wav'
wavfilename2 = 'D:\git-repo\MFA_DAE_ME\src\dataset\224_1b1_Tc_sc_Meditron.wav'

FontSize = 20
figure(1)
% p = abs(fft(periodic_feature(20:end,6111)))
subplot(121)
[CleanSpec,sig] = wav2spec(wavfilename1);

% h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
h_pc=imagesc(sqrt(CleanSpec/std(sig)));
colormap Jet;axis xy;

% xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
set(gca, 'CLim', [1,30]);%frequency band
set(gca, 'YTick',[0,65,129]);
set(gca,'YTickLabel',{'0','500','1000'})
set(gca, 'XTick',[1,250,494]);
set(gca,'XTickLabel',{'0','5','10'})
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
  xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');

% set (gca,'position',[0.1,0.3,0.3,0.5] );%?�m��??�bfigture����?�Z,�|???��?��ܪ��O��?�ɡA�U?�ɡA���סA?�סC
set(gcf,'Position',[100 100 260 220]);

subplot(122)
[CleanSpec,sig] = wav2spec(wavfilename2);
% h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
h_pc=imagesc(sqrt(CleanSpec/std(sig)));
size(sqrt(CleanSpec/std(sig)))
colormap Jet;axis xy;box off;

set(gca, 'CLim', [1,32]);%frequency band

set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
set(gca, 'YTick',[0,65,129]);
set(gca,'YTickLabel',{'0','500','1000'})
set(gca, 'XTick',[1,250,494]);
set(gca,'XTickLabel',{'0','5','10'})
  xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');

% ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
% set (gca,'position',[0.5,0.3,0.3,0.5] );
% set(gcf,'unit','normalized','position',[0.2,0.2,0.48,0.32]);
set(gcf,'Position',[100 100 260 220]);
set(gcf,'unit','normalized','position',[0.2,0.2,0.58,0.26]);
% set(gcf,'unit','normalized','position',[0.2,0.2,0.9,0.32]);
