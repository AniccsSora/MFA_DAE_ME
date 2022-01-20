clear;
FrameLength=256;
FrameRate=85;
FFT_SIZE=FrameLength;
flag=1;

sr=8000;

shift=0;
shrink=0;
exp_path = '.\exp_fig\';

%---- ���|�]�w
% home_windows ='D:\Git'
% school_windows='D:\git-repo'
prj_path="D:\git-repo";  % �M�צ�m  //linux : /home/user/git-repo,  
% ---------------------------------------------------------------
timestamp_foldern='DAE_C_2022_0120_1318_27'; % log �U ���W��Ƨ��W
%
origin_path=fullfile(prj_path,'MFA_DAE_ME','src','dataset'); % �L�l source �ڥؿ�
% ���ث᭵��
recons_path=fullfile(prj_path,'MFA_DAE_ME','src','log',timestamp_foldern,'test_reconstruct');
s1_path=fullfile(prj_path,'MFA_DAE_ME','src','log',timestamp_foldern,'test_source1'); % test_my_source1, test_source1
s2_path=fullfile(prj_path,'MFA_DAE_ME','src','log',timestamp_foldern,'test_source2'); % test_my_source2, test_source2
s3_path=fullfile(prj_path,'MFA_DAE_ME','src','log',timestamp_foldern,'test_source3');
lab_dir_name='';  % ��� python ���� lab_dir_name ��Ƨ��ܼơA�p�S�]�w�i�H����

origin=fullfile(origin_path, '4_1.wav'); % ���إX�Ӫ� wav name
recons=fullfile(recons_path, lab_dir_name, '4_1.wav');
source1 =fullfile(s1_path, lab_dir_name,'4_1.wav');
source2 =fullfile(s2_path, lab_dir_name,'4_1.wav');
source3 =fullfile(s3_path, lab_dir_name,'4_1.wav');

%%% �������|�A�ϵ��ѡC
% recons= "/home/user/git-repo/MFA_DAE_20210904_fft/test/_wavresult/reconstruct/test.wav";
% source1="/home/user/git-repo/MFA_DAE_20210904_fft/test/_wavresult/source1/test.wav";
% source2 ="/home/user/git-repo/MFA_DAE_20210904_fft/test/_wavresult/source2/test.wav";
% source3 ="/home/user/git-repo/MFA_DAE_20210904_fft/test/_wavresult/reconstruct/test.wav";

source4 = 'D:\Git\qpzm_mfa_dae\src\log\MFA_ANA\test_source4\0_0_w_off\4_0_0_old_method_����wienner_mask_PR20.wav';

my_fig_title='4\_1'; % use class 3

FontSize = 18;
%  [0.01 0.07 0.98 0.82] �̨οù����X��ܮĪG
figure('visible','on','units','normalized','Position', [0.0,0.1,0.97,0.75]);  % ���U(���U��0) ������m(a b)�B (�e ��)
%---------------------------------------------------------
% ø�s��l���W��
subplot(231);
[CleanSpec,sig] = wav2spec(origin);
%h_pc=imagesc(sqrt(CleanSpec(1:65,:)/std(sig))); %���W���W��? 1000~2000
h_pc=imagesc(sqrt(CleanSpec(1:129,:)/std(sig))); %���W���W��? 1000~2000
title(gca,"origin");
colormap Jet;axis xy;

set(gca, 'CLim', [1,119]);%frequency band
% �W�� y�b 
set(gca, 'YTick',[0,64,129]);
set(gca,'YTickLabel',{'0','500','1000'});
set(gca, 'XTick',[1,2468,4936]);
set(gca,'XTickLabel',{'0','5','10'});
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
%xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
%set(gcf,'Position',[100 100 260 220]);
%---------------------------------------------------------
% ø�s���ت��W��
subplot(232);
[CleanSpec,sig] = wav2spec(recons);
h_pc=imagesc(sqrt(CleanSpec(1:129,:)/std(sig))); %���W���W��?
title(gca,"reconstruction");
colormap Jet;axis xy;

set(gca, 'CLim', [1,120]);%frequency band
% �W�� y�b 
set(gca, 'YTick',[0,64,129]);
set(gca,'YTickLabel',{'0','500','1000'});
set(gca, 'XTick',[1,2468,4936]);
set(gca,'XTickLabel',{'0','5','10'});
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
%ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
%xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
%set(gcf,'Position',[10 10 900 600]);
%---------------------------------------------------------
subplot(234);
[CleanSpec,sig] = wav2spec(source1);
h_pc=imagesc(sqrt(CleanSpec(1:129,:)/std(sig))); %���W���W��?
title(gca,"source 1");
colormap Jet;axis xy;

set(gca, 'CLim', [1,120]);%frequency band
% �W�� y�b 
set(gca, 'YTick',[0,64,129]);
set(gca,'YTickLabel',{'0','500','1000'});
set(gca, 'XTick',[1,2468,4936]);
set(gca,'XTickLabel',{'0','5','10'});
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
%set(gcf,'Position',[100 100 260 220]);
%---------------------------------------------------------
subplot(235);
[CleanSpec,sig] = wav2spec(source2);
h_pc=imagesc(sqrt(CleanSpec(1:129,:)/std(sig)));
title(gca,"source 2");
size(sqrt(CleanSpec/std(sig)));
colormap Jet;axis xy;box off;

set(gca, 'CLim', [1,120]);%frequency band

set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
set(gca, 'YTick',[0,64,129]);
set(gca,'YTickLabel',{'0','500','1000'});
set(gca, 'XTick',[1,2468,4936]);
set(gca,'XTickLabel',{'0','5','10'});
xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
%set(gca,'OuterPosition', [0.5,0.1,0.5,0.47]);
%fprintf('4: %1.2f %1.2f %1.2f %1.2f\n',get(gca,'OuterPosition'));
%set(gca,'Position')%  [left bottom width height]

%set(gcf,'Position',[100 100 260 220]);
%set(gcf,'unit','normalized','Position',[0.2,0.2,0.58,0.26]);
%---------------------------------------------------------
subplot(236);
[CleanSpec,sig] = wav2spec(source3);
h_pc=imagesc(sqrt(CleanSpec(1:129,:)/std(sig)));
title(gca,"source 3");
size(sqrt(CleanSpec/std(sig)));
colormap Jet;axis xy;box off;

set(gca, 'CLim', [1,120]);%frequency band

set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
set(gca, 'YTick',[0,64,129]);
set(gca,'YTickLabel',{'0','500','1000'});
set(gca, 'XTick',[1,2468,4936]);
set(gca,'XTickLabel',{'0','5','10'});
xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
%set(gca,'OuterPosition', [0.5,0.1,0.5,0.47]);
%fprintf('4: %1.2f %1.2f %1.2f %1.2f\n',get(gca,'OuterPosition'));
%set(gca,'Position')%  [left bottom width height]

%set(gcf,'Position',[100 100 2600 220]);
%set(gcf,'unit','normalized','Position',[0.2,0.2,0.58,0.26]);
%---------------------------------------------------------
% source num = 4
% subplot(233);
% [CleanSpec,sig] = wav2spec(source4);
% h_pc=imagesc(sqrt(CleanSpec(1:65,:)/std(sig)));
% title(gca,"source 4");
% size(sqrt(CleanSpec/std(sig)));
% colormap Jet;axis xy;box off;
% 
% set(gca, 'CLim', [1,120]);%frequency band
% 
% set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
% set(gca, 'YTick',[0,32,65]);
% set(gca,'YTickLabel',{'0','1000','2000'});
% set(gca, 'XTick',[1,995,1994]);
% set(gca,'XTickLabel',{'0','5','10'});
% %xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
% set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
% %set(gca,'OuterPosition', [0.5,0.1,0.5,0.47]);
% %fprintf('4: %1.2f %1.2f %1.2f %1.2f\n',get(gca,'OuterPosition'));
% %set(gca,'Position')%  [left bottom width height]
% 
% set(gcf,'Position',[100 100 260 220]);
% set(gcf,'unit','normalized','Position',[0.2,0.2,0.58,0.26]);
%---------------------------------------------------------
sgt=sgtitle(my_fig_title);
sgt.FontSize=18;
sgt.FontName='Times New Roman';
sgt.FontWeight='bold'; % normal | bold

f = gcf; 
exportgraphics(f,'Output_filename.png','Resolution', 200);