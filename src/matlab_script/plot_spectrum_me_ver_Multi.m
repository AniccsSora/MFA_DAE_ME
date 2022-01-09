clear;
FrameLength=256;
FrameRate=40;
FFT_SIZE=FrameLength;
flag=1;

sr=8000;

shift=0;
shrink=0;
exp_path = '.\exp_fig\';

%---- 路徑設定
% prj_path='D:\Git';  % 專案位置
% origin_path=fullfile(prj_path,'qpzm_MFA_DAE\src');
% recons_path=fullfile(prj_path,'qpzm_MFA_DAE\src\log\MFA_ANA\test_reconstruct');
% s1_path=fullfile(prj_path,'qpzm_MFA_DAE\src\log\MFA_ANA\test_source1');
% s2_path=fullfile(prj_path,'qpzm_MFA_DAE\src\log\MFA_ANA\test_source2');
% s3_path=fullfile(prj_path,'qpzm_MFA_DAE\src\log\MFA_ANA\test_source3');
% lab_dir_name='多筆資料測試';  % 比照 python 內的 lab_dir_name 資料夾變數，如沒設定可以為空
% 
% origin=fullfile(origin_path, '5_1.wav');
% recons=fullfile(recons_path, lab_dir_name, '0_5_1_old_method_PR13.wav');
% source1=fullfile(s1_path, lab_dir_name,'1_5_1_old_method_PR13.wav');
% source2 =fullfile(s2_path, lab_dir_name,'2_5_1_old_method_PR13.wav');
% source3 =fullfile(s3_path, lab_dir_name,'3_5_1_old_method_PR13.wav');

% 原始 音訊
% 重跑記得改 
% D:\Git\qpzm_mfa_dae\src\dataset\training_noisy_心肺\0dB
% D:\Git\qpzm_mfa_dae\src\dataset\audio_Healthy\*.wav
originList=getFilelistByPattern('D:\Git\qpzm_mfa_dae\src\dataset\audio_Healthy\*.wav'); % 這裡面的 imgSaveName 醜醜的不用管他
reconsList=getFilelistByPattern("D:\Git\qpzm_mfa_dae\src\log\MFA_ANA\test_reconstruct\old_model_healthy_audio_disPR_5_wienner_mask_True\*old*");
source1List=getFilelistByPattern("D:\Git\qpzm_mfa_dae\src\log\MFA_ANA\test_source1\old_model_healthy_audio_disPR_5_wienner_mask_True\*old*");
source2List=getFilelistByPattern("D:\Git\qpzm_mfa_dae\src\log\MFA_ANA\test_source2\old_model_healthy_audio_disPR_5_wienner_mask_True\*old*");
source3List=getFilelistByPattern("D:\Git\qpzm_mfa_dae\src\log\MFA_ANA\test_source3\old_model_healthy_audio_disPR_5_wienner_mask_True\*old*");
% 使用  *old* 或是 *use_class3*
FOR_DEBUG_VAR = {originList, reconsList, source1List, source2List, source3List};

% 圖片的另存資料夾
as_save_folderName='old_model_healthy_audio_disPR_5_wienner_mask_True';

% 目標資料夾們
for ii = 1:length(reconsList)
    origin= originList(ii).fullpath;
    recons= reconsList(ii).fullpath;
    source1=source1List(ii).fullpath;
    source2 =source2List(ii).fullpath;
    source3 =source3List(ii).fullpath;

    % recons= "D:\Git\MFA_DAE\src\log\default\DAE_C_20210901_1611\test_reconstruct\5_1.wav";
    % source1="D:\Git\MFA_DAE\src\log\default\DAE_C_20210901_1611\test_source1\5_1.wav";
    % source2 ="D:\Git\MFA_DAE\src\log\default\DAE_C_20210901_1611\test_source2\5_1.wav";
    % source3 ="D:\Git\MFA_DAE\src\log\default\DAE_C_20210901_1611\test_source3\5_1.wav";

    my_fig_title=reconsList(ii).imgSaveName; % 圖片標題

    FontSize = 18;
    %  [0.01 0.07 0.98 0.82] 最佳螢幕跳出顯示效果
    figure('visible','off','units','normalized','Position', [0 0 1 1]);  % 左下(左下為0) 視窗位置(a b)、 (寬 高)
    %---------------------------------------------------------
    % 繪製原始的頻譜
    subplot(231);
    [CleanSpec,sig] = wav2spec(origin);
    h_pc=imagesc(sqrt(CleanSpec(1:65,:)/std(sig))); %正規化頻譜?
    title(gca,"origin");
    colormap Jet;axis xy;

    set(gca, 'CLim', [1,120]);%frequency band
    % 頻譜 y軸 
    set(gca, 'YTick',[0,32,65]);
    set(gca,'YTickLabel',{'0','1000','2000'});
    set(gca, 'XTick',[1,995,1994]);
    set(gca,'XTickLabel',{'0','5','10'});
    set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
    ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
    %xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
    %set(gcf,'Position',[100 100 260 220]);
    %---------------------------------------------------------
    % 繪製重建的頻譜
    subplot(232);
    [CleanSpec,sig] = wav2spec(recons);
    h_pc=imagesc(sqrt(CleanSpec(1:65,:)/std(sig))); %正規化頻譜?
    title(gca,"reconstruction");
    colormap Jet;axis xy;

    set(gca, 'CLim', [1,120]);%frequency band
    % 頻譜 y軸 
    set(gca, 'YTick',[0,32,65]);
    set(gca,'YTickLabel',{'0','1000','2000'});
    set(gca, 'XTick',[1,995,1994]);
    set(gca,'XTickLabel',{'0','5','10'});
    set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
    %ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
    %xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
    %set(gcf,'Position',[10 10 900 600]);
    %---------------------------------------------------------
    subplot(234);
    [CleanSpec,sig] = wav2spec(source1);
    h_pc=imagesc(sqrt(CleanSpec(1:65,:)/std(sig))); %正規化頻譜?
    title(gca,"source 1");
    colormap Jet;axis xy;

    set(gca, 'CLim', [1,120]);%frequency band
    % 頻譜 y軸 
    set(gca, 'YTick',[0,32,65]);
    set(gca,'YTickLabel',{'0','1000','2000'});
    set(gca, 'XTick',[1,995,1994]);
    set(gca,'XTickLabel',{'0','5','10'});
    set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
    ylabel('Frequency (Hz)','fontsize',FontSize,'Fontname','Times New Roman');
    xlabel({'Time(sec)'},'fontsize',FontSize,'Fontname','Times New Roman');
    %set(gcf,'Position',[100 100 260 220]);
    %---------------------------------------------------------
    subplot(235);
    [CleanSpec,sig] = wav2spec(source2);
    h_pc=imagesc(sqrt(CleanSpec(1:65,:)/std(sig)));
    title(gca,"source 2");
    size(sqrt(CleanSpec/std(sig)));
    colormap Jet;axis xy;box off;

    set(gca, 'CLim', [1,120]);%frequency band

    set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
    set(gca, 'YTick',[0,32,65]);
    set(gca,'YTickLabel',{'0','1000','2000'});
    set(gca, 'XTick',[1,995,1994]);
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
    h_pc=imagesc(sqrt(CleanSpec(1:65,:)/std(sig)));
    title(gca,"source 3");
    size(sqrt(CleanSpec/std(sig)));
    colormap Jet;axis xy;box off;

    set(gca, 'CLim', [1,120]);%frequency band

    set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
    set(gca, 'YTick',[0,32,65]);
    set(gca,'YTickLabel',{'0','1000','2000'});
    set(gca, 'XTick',[1,995,1994]);
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
    my_fig_title=strrep(my_fig_title,'_','\_'); % 畫在 figure 上的 "_" 字元要設為逃脫字元。
    sgt=sgtitle(my_fig_title);
    sgt.FontSize=18;
    sgt.FontName='Times New Roman';
    sgt.FontWeight='bold'; % normal | bold

    f = gcf; 
    save_image_name=sprintf('spec_%s.png', reconsList(ii).imgSaveName);
    [ ~, ~ ] = mkdir(as_save_folderName);  % 忽略輸出
    exportgraphics(f, strcat('./',as_save_folderName,'/',save_image_name), 'Resolution', 200);
end