clear;
%FrameLength=256;
%FrameRate=40;
%FFT_SIZE=FrameLength;
%flag=1;

%sr=8000;

shift=0;
shrink=0;
exp_path = '.\exp_fig\';

% home
% wavfilename1 = 'D:\git-repo\qpzm_MFA_DAE\src\senpai_data\heart_lung_sam2\mix\training_clean_心跳\0dB\4_1.wav';
% wavfilename2 = 'D:\git-repo\qpzm_MFA_DAE\src\senpai_data\heart_lung_sam2\mix\training_noise_呼吸\0dB\4_1.wav';

% school
% wavfilename1 = 'D:\git-repo\qpzm_mfa_dae\src\senpai_data\heart_lung_sam2\mix\training_clean_心跳\0dB\4_1.wav';
% wavfilename2 = 'D:\git-repo\qpzm_mfa_dae\src\senpai_data\heart_lung_sam2\mix\training_noise_呼吸\0dB\4_1.wav';

% 隨意測試
wavfilename1 = 'D:\git-repo\qpzm_mfa_dae\src\senpai_data\heart_lung_sam2\mix\training_clean_心跳\0dB\4_1.wav';
wavfilename2 = 'D:\git-repo\qpzm_mfa_dae\src\senpai_data\heart_lung_sam2\mix\training_noise_呼吸\0dB\4_1.wav';

FontSize = 18;
figure(1);
% p = abs(fft(periodic_feature(20:end,6111)))
subplot(121);
[CleanSpec,sig] = wav2spec(wavfilename1);
% [debug_CleanSpec, debug_sig] = wav2spec(debug_wavfilename1);

% h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
% 原本是使用 1:65
h_pc=imagesc(sqrt(CleanSpec(1:129,:)/std(sig)));% 總高的 1/4 
colormap Jet;axis xy;

% xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
set(gca, 'CLim', [1,100]);%frequency band  
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 頻譜 y軸 
% ~原本的
%set(gca, 'YTick',[0,32,65]);
% set(gca,'YTickLabel',{'0','1000','2000'})
set(gca, 'YTick',[0,64,129]);
set(gca,'YTickLabel',{'0','500','1000'});
% set(gca, 'XTick',[1,250,494]);
set(gca, 'XTick',[1,995,1994]);
set(gca,'XTickLabel',{'0','5','10'});
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
  xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');

% set (gca,'position',[0.1,0.3,0.3,0.5] );%?置坐??在figture中的?距,四???分?表示的是左?界，下?界，高度，?度。
set(gcf,'Position',[100 100 260 220]);

subplot(122);
[CleanSpec,sig] = wav2spec(wavfilename2);
% h_pc=imagesc((shift+1:(size(CleanSpec,2)-shrink))*FrameRate/sr,0:sr/2/(size(CleanSpec,1)-1):sr/2,sqrt(CleanSpec/std(sig)));
h_pc=imagesc(sqrt(CleanSpec(1:129,:)/std(sig)));
size(sqrt(CleanSpec/std(sig)));
colormap Jet;axis xy;box off;

set(gca, 'CLim', [1,120]);%frequency band

set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
% set(gca, 'YTick',[0,65,129]);
% set(gca,'YTickLabel',{'0','1000','4000'})
set(gca, 'YTick',[0,64,129]);
set(gca,'YTickLabel',{'0','500','1000'});
% set(gca, 'XTick',[1,250,494]);
set(gca, 'XTick',[1,995,1994]);
set(gca,'XTickLabel',{'0','5','10'})
xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');

% ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
% set (gca,'position',[0.5,0.3,0.3,0.5] );
% set(gcf,'unit','normalized','position',[0.2,0.2,0.48,0.32]);
set(gcf,'Position',[100 100 260 220]);
set(gcf,'unit','normalized','position',[0.0,0.5,0.71,0.36]); % 'Position', [0 0 1 1]  [0.2,0.2,0.58,0.26]
% set(gcf,'unit','normalized','position',[0.2,0.2,0.9,0.32]);
