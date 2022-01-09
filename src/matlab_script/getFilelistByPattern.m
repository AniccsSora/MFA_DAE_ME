function [result] = getFilelistByPattern(fn_pattern)
%{
取得某資料夾下的檔案名稱
%}
result=struct();

flist=dir(fn_pattern);
    for i = 1:numel(flist)
           result(i).fullpath=fullfile(flist(i).folder, flist(i).name);
           
           % 分檔名出來
           tmp_1=split(flist(i).name, '.');
           result(i).name=tmp_1(1,:);
           
           % 掠過檔名前兩個字元 (存圖片用的檔名)
           tmp_2=result(i).name{1};
           result(i).imgSaveName=tmp_2(3:length(tmp_2));
    end
end

