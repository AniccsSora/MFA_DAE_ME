clear;
FrameLength=256;
FrameRate=40;
FFT_SIZE=FrameLength;
flag=1;

sr=2000;

shift=0;
shrink=0;

exp_path = 'D:\exp_fig\';
wavfilename1 = 'D:\Git\qpzm_mfa_dae\src\dataset\training_noisy_心肺\0dB\4_0.wav';
wavfilename2 = 'D:\Git\qpzm_mfa_dae\src\dataset\training_noisy_心肺\0dB\4_1.wav';

FontSize = 10
figure(1)
% p = abs(fft(periodic_feature(20:end,6111)))
subplot(121)
    audioData1 = audioread(wavfilename1);
    sr = 8000
LineWidth=1;
h0 = plot(0:1/sr:(length(audioData1)-1)/sr,(audioData1)/std(audioData1));

set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
set(gca, 'YLim',[-10,10]);
%  xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
set(gca, 'CLim', [1,30]);%frequency band
ylabel('Amplitude','fontsize',FontSize,'Fontname','Times New Roman');

% set (gca,'position',[0.1,0.3,0.3,0.5] );%?置坐??在figture中的?距,四???分?表示的是左?界，下?界，高度，?度。
set(gcf,'Position',[100 100 260 220]);

subplot(122)
    audioData2 = audioread(wavfilename2);
    sr = 8000
LineWidth=1;
h0 = plot(0:1/sr:(length(audioData2)-1)/sr,(audioData2)/std(audioData2));

set(gca, 'CLim', [1,32]);%frequency band

set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
set(gca, 'YLim',[-10,10]);
%    xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
% set (gca,'position',[0.5,0.3,0.3,0.5] );
% set(gcf,'unit','normalized','position',[0.2,0.2,0.48,0.32]);
set(gcf,'Position',[100 100 260 220]);
set(gcf,'unit','normalized','position',[0.2,0.2,0.58,0.32]);

% set(gcf,'unit','normalized','position',[0.2,0.2,0.9,0.32]);
