
ans = load('D:\1file_heart_lung_sam2_1219_3\HL_FFT_fig\-2dB\5_1.wav.mat')
f_class_label = ans.class_label;
f_periodic_feature = ans.periodic_feature;
ans =load('D:\1file_heart_lung_sam2_1219_3\HL_fig\-2dB\5_1.wav.mat')
b_class_label = ans.class_label;
b_feature = ans.periodic_feature;
%  aa = fft(smooth(b_feature(:,neuron_num ))-mean(smooth(b_feature(:,neuron_num ))))/size(smooth(b_feature(:,neuron_num )),2); 
%      aa = abs(aa(1:floor(length(aa)/2+1)));
%      period_feature(:,1) = aa; 
%      diff_period(1,1) = max(aa(2:end))-median(aa);

neuron_num = 198
neuron_num = 4426
neuron_num2 = 3
disp(f_class_label(neuron_num))
figure(1)
% p = abs(fft(periodic_feature(20:end,6111)))
% plot(smooth(b_feature(:,neuron_num )),'LineWidth',2) 
 plot(b_feature(:,neuron_num ),'LineWidth',1.5) 
% fs=8000;		% Sampling rate
% filterOrder=4;		% Order of filter
% [b, a]=butter(filterOrder, [10,1300]/(fs/2));
% f_periodic_feature(1:50,neuron_num )=filter(b, a, f_periodic_feature(1:50,neuron_num ));
 figure(2) %�w?�@??3,3�O?���W?
plot(f_periodic_feature(1:100,neuron_num ),'LineWidth',1.5);
f_periodic_feature(1:100,neuron_num ) = smooth(f_periodic_feature(1:100,neuron_num )) ;
 figure(3) %�w?�@??3,3�O?���W?
plot(f_periodic_feature(1:100,neuron_num ),'LineWidth',1.5);
% 
% figure(1) %�w?�@??3,3�O?���W?
% subplot(4,1,1) % ??3�E��?1��2�C�A�w��b��1��
% plot(f_periodic_feature(:,neuron_num ))       %�b?��?bart?
% title(f_class_label(neuron_num))
% subplot(4,1,2) %??3�E��?1��2�C, �w��b��2��
% plot(b_feature(:,neuron_num))       % �b?��?hist?
% title(b_class_label(neuron_num))
% subplot(4,1,3) %??3�E��?1��2�C, �w��b��2��
% plot(f_periodic_feature(:,neuron_num2))       % �b?��?hist?
% title(f_class_label(neuron_num2))
% subplot(4,1,4) %??3�E��?1��2�C, �w��b��2��
% plot(b_feature(:,neuron_num2))       % �b?��?hist?
% title(b_class_label(neuron_num2))
% % plot(period_feature(:,1))