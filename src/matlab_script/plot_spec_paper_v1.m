
ans = load('D:\1file_heart_lung_sam2_1219_3\HL_FFT_fig\-2dB\5_1.wav.mat')
f_class_label = ans.class_label;
f_periodic_feature = ans.periodic_feature;
ans =load('D:\1file_heart_lung_sam2_1219_3\HL_fig\-2dB\5_1.wav.mat')
b_class_label = ans.class_label;
b_feature = ans.periodic_feature;

% 
%   aa = fft(smooth(b_feature(:,neuron_num ))-mean(smooth(b_feature(:,neuron_num ))))/size(smooth(b_feature(:,neuron_num )),2); 
%   aa = abs(aa(1:floor(length(aa)/2+1)));
%  period_feature(:,1) = aa; 
%  diff_period(1,1) = max(aa(2:end))-median(aa);


neuron_num = 7215%lung7215(-2dB_5_1)
neuron_num2 = 2115 %heart2115(6db_4_1)
disp(f_class_label(neuron_num))
FontSize = 12
figure(1)
% p = abs(fft(periodic_feature(20:end,6111)))
subplot(222)
plot(smooth(b_feature(1:600,neuron_num )),'LineWidth',2) 

 set(gca, 'XTick',[0,300, 600]);
 set(gca,'XTickLabel',{'0','300','600'})
% set (gca,'position',[0.1,0.3,0.3,0.5] );
set(gcf,'Position',[100 100 260 220]);
%  set(gcf,'unit','normalized','position',[0.2,0.2,0.56,0.32]);
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel({'Frame','(d)'},'fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Latent value (Relu)','fontsize',FontSize,'Fontname','Times New Roman');
title=('Lung');
subplot(224)
plot(f_periodic_feature(1:100,neuron_num ),'LineWidth',2) 
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel({'Frequency (Preiodicity)','(f)'},'fontsize',FontSize,'Fontname','Times New Roman');
% ylabel('Power spectrum density','fontsize',FontSize,'Fontname','Times New Roman');
title=('Lung');
% set (gca,'position',[0.5,0.3,0.3,0.5] );
% set(gcf,'unit','normalized','position',[0.2,0.2,0.48,0.32]);
set(gcf,'Position',[100 100 260 220]);
 set(gcf,'unit','normalized','position',[0.2,0.2,0.56,0.32]);


ans = load('D:\1file_heart_lung_sam2_1219_3\HL_FFT_fig\6dB\4_1.wav.mat')
f_class_label = ans.class_label;
f_periodic_feature = ans.periodic_feature;
ans =load('D:\1file_heart_lung_sam2_1219_3\HL_fig\6dB\4_1.wav.mat')
b_class_label = ans.class_label;
b_feature = ans.periodic_feature;
neuron_num = 7215%lung7215(-2dB_5_1)
neuron_num2 = 2117 %heart2115(6db_4_1)
% p = abs(fft(periodic_feature(20:end,6111)))
subplot(221)
plot(smooth(b_feature(1:600,neuron_num2 )),'LineWidth',2) 
title=('Lung');
set(gca, 'XTick',[0,300, 600]);
set(gca,'XTickLabel',{'0','300','600'})
set(gcf,'Position',[100 100 260 220]);

% set(gca,'Position',[.13 .17 .80 .74]);
% set (gca,'position',[0.1,0.3,0.3,0.5] );%?置坐??在figture中的?距,四???分?表示的是左?界，下?界，高度，?度。
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel({'Frame','(c)'},'fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Latent value (ReLU)','fontsize',FontSize,'Fontname','Times New Roman');

ax3=subplot(223)
plot(f_periodic_feature(1:100,neuron_num2 ),'LineWidth',2) 
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
xlabel({'Frequency (Preiodicity)','(e)'},'fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Power spectrum density','fontsize',FontSize,'Fontname','Times New Roman');
{'Population','(in thousands)'}
%  set(gcf,'unit','normalized','position',[0.2,0.2,0.48,0.32]);
set(gcf,'Position',[100 100 260 220]);
set(gcf,'unit','normalized','position',[0.2,0.2,0.56,0.64]);