
FrameLength=256;
FrameRate=40;
FFT_SIZE=FrameLength;
flag=1;

sr=8000;

shift=0;
shrink=0;
wavfilename1 = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source0\6dB\4_1.wav'
wavfilename2 = 'D:\import_data\heart_lung_sam2_exp_eval\heart_lung_sam2\mix\source1\-2dB\5_1.wav'
FontSize = 12
figure(1)
% p = abs(fft(periodic_feature(20:end,6111)))
subplot(121)
[CleanSpec,sig] = wav2spec(wavfilename1);

h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
size(sqrt(CleanSpec/std(sig)))
colormap Jet;axis xy;box off;

set(gca, 'CLim', [1 30]);%frequency band
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
% set(gca, 'XTick',[1:1000:1500]);
% set(gca,'XTickLabel',{'0','5','10'});
% set(gca, 'YTick',[1:60:129]);
% set(gca,'YTickLabel',{'0','1000','2000'});
%   set(gca, 'XTick',[1:1000:1994]);
%   set(gca,'XTickLabel',{'0','5','10'});

xlabel({'Time(sec)','(a)'},'fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');

% set (gca,'position',[0.1,0.3,0.3,0.5] );%?置坐??在figture中的?距,四???分?表示的是左?界，下?界，高度，?度。
set(gcf,'Position',[100 100 260 220]);

subplot(122)
[CleanSpec,sig] = wav2spec(wavfilename2);
h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
colormap Jet;axis xy;box off;
set(gca, 'CLim', [1 30]);%frequency band
%   set(gca, 'YTick',[1:60:129]);
%   set(gca,'YTickLabel',{'0','1000','2000'});
%   set(gca, 'XTick',[1:1000:1994]);
%   set(gca,'XTickLabel',{'0','300','600'});
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel({'Time(sec)','(b)'},'fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
% set (gca,'position',[0.5,0.3,0.3,0.5] );
% set(gcf,'unit','normalized','position',[0.2,0.2,0.48,0.32]);
set(gcf,'Position',[100 100 260 220]);
set(gcf,'unit','normalized','position',[0.2,0.2,0.56,0.32]);
