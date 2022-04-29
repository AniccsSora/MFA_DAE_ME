clear;
FrameLength=256;
FrameRate=40;
FFT_SIZE=FrameLength;
flag=1;

sr=2000;

shift=0;
shrink=0;

  wavfilename1 =  'D:\git-repo\MFA_DAE_ME\src\dataset\224_1b1_Tc_sc_Meditron.wav'; % �ɰ��
%  wavfilename2 = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source1\6dB\4_1.wav'
%

FontSize = 12
figure(1)
% p = abs(fft(periodic_feature(20:end,6111)))
    audioData2 = audioread(wavfilename1);
    sr = 8000
LineWidth=1;
h0 = plot(0:1/sr:(length(audioData2)-1)/sr,(audioData2)/std(audioData2));

set(gca, 'CLim', [1,32]);%frequency band

set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
set(gca, 'YLim',[-10,10]);
set(gca, 'CLim', [1 30]);%frequency band
%  set(gca, 'YTick',[1:60:129]);
%   set(gca,'YTickLabel',{'0','1000','2000'});
%   set(gca, 'XTick',[1:250:494]);
%   set(gca,'XTickLabel',{'0','300','600'});

set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
%    xlabel({'Frame'},'fontsize',FontSize,'Fontname','Times New Roman');

ylabel('Amplitude','fontsize',FontSize,'Fontname','Times New Roman');

% set (gca,'position',[0.1,0.3,0.3,0.5] );%?�m��??�bfigture����?�Z,�|???��?��ܪ��O��?�ɡA�U?�ɡA���סA?�סC
%  set(gcf,'unit','normalized','position',[0.2,0.2,0.48,0.32]);
set(gcf,'Position',[100 100 260 220]);
set(gcf,'unit','normalized','position',[0.2,0.2,0.56,0.64]);

