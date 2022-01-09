%音訊名稱
audio_name="./0_0.wav";

% 讀檔 轉頻譜圖



[CleanSpec, sig]=wav2spec(audio_name);

origin_wave=audioread(audio_name);

subplot(1,2,1);
plot(sig);

subplot(1,2,2);
plot((origin_wave+eps));
