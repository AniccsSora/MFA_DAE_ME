import librosa
import scipy.io.wavfile as wav
import numpy as np

if __name__ == "__main__":
    # 分離出來的聲音
    s1 = "./log/DAE_C_2022_0109_2149_01/test_source1/4_1.wav"
    s2 = "./log/DAE_C_2022_0109_2149_01/test_source2/4_1.wav"
    s3 = "./log/DAE_C_2022_0109_2149_01/test_source3/4_1.wav"
    s1, s2, s3 = wav.read(s1)[1], wav.read(s2)[1], wav.read(s3)[1]
    s1, s2 = s1.astype('float64'), s2.astype('float64')
    s3 = s3.astype('float64')
    # 純音
    p1 = r"./dataset/senpai_data/heart_lung_sam2/mix/training_clean_心跳/0dB/4_1.wav"
    p2 = r"./dataset/senpai_data/heart_lung_sam2/mix/training_noise_呼吸/0dB/4_1.wav"
    p1, p2 = wav.read(p1)[1], wav.read(p2)[1]
    p1, p2 = p1.astype('float64'), p2.astype('float64')

    print("sep1 pure1 差異 square error:\t", np.sum(np.sqrt(np.square(s1 - p1)), axis=0))
    print("sep2 pure2 差異 square error:\t", np.sum(np.sqrt(np.square(s2 - p2)), axis=0))
    print("sep2 sep3 差異 square error:\t", np.sum(np.sqrt(np.square(s2 - s3)), axis=0))

    print("sep1 sep2 差異 square error: \t", np.sum(np.sqrt(np.square(s1 - s2)), axis=0))
    print("pure1 pure2 差異 square error:\t", np.sum(np.sqrt(np.square(p1 - p2)), axis=0))

