import numpy as np
from torch.utils.data import DataLoader

from dataset.HLsep_dataloader import hl_dataloader
from typing import List, Dict, Any, AnyStr
import torch
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
from tqdm import tqdm
import multiprocessing
import cv2

def analysis_plot_neuron_representation(_1d_tensor, save_name, audio_total_sec, audio_during_first_sec):
    # 因為全長 10秒 作分析太多了故 擷取部分長度即可
    audio_total = audio_total_sec  # 音源全長
    audio_take_first = audio_during_first_sec  # 開頭前 n 秒

    # 計算 擷取占比
    _afp = audio_take_first/audio_total

    _1d_tensor_log10 = np.log10(_1d_tensor)

    # 截斷後長度
    cut_length = int(len(_1d_tensor) * _afp)
    # 取得截斷版本
    _1d_tensor_short = _1d_tensor[0:cut_length]
    _1d_tensor_short_log10 = _1d_tensor_log10[0:cut_length]

    ax = plt.subplot(221)
    ax.set_title("latent representation")
    plt.plot(_1d_tensor_short)
    ax = plt.subplot(223)
    ax.set_title("latent representation (log10)")
    plt.plot(_1d_tensor_short_log10)

    ax = plt.subplot(222)
    #ax.set_title("fft latent representation")
    _1d_tensor_short_odd_minus1 = np.copy(_1d_tensor_short)

    # 奇數idx 值*-1
    # for idx, val in enumerate(_1d_tensor_short_odd_minus1):
    #     if idx%2 == 0 :
    #         pass
    #     else:
    #         _1d_tensor_short_odd_minus1[idx] *= -1.0
    # _ = np.abs(np.fft.fft(_1d_tensor_short_odd_minus1))  # 做一維傅立葉變換 奇數idx 值*-1

    _ = np.abs(np.fft.fft(_1d_tensor_short))  # 做一維傅立葉變換
    assert _.ndim == 1
    _freq = np.fft.fftfreq(_.size, d=8000)
    # _ = np.log10(_)  # ------------------------------------  取 log
    draw_val = _[1:len(_) // 2 + 1]  # 不使用 fft 第 0 個的數值
    plt.plot(draw_val)
    pick_vline = np.argmax(draw_val)
    plt.axvline(x=pick_vline, color='red', linewidth=1)
    ax.set_title(f"fft latent representation, p={pick_vline}")

    ax = plt.subplot(224)
    ax.set_title("fft latent representation (log10)")
    _ = np.abs(np.fft.fft(_1d_tensor_short_log10))
    # _ = np.log10(_)  # ------------------------------------  取 log
    plt.plot(_[1:len(_)//2+1])  # 不使用 fft 第 0 個的數值

    #
    plt.tight_layout()

    #
    dir_name = './latent_code_fft_analysis'
    os.makedirs(dir_name, exist_ok=True)

    save_name = pjoin(dir_name, save_name)
    plt.savefig(save_name)
    plt.close()
    plt.cla()
    plt.clf()


def do_latent_representation_fft(node_representation, remove_first=True):
    """
    Args:
        node_representation: latent space matrix.
        remove_first: (default=True) 捨去fft後的第0個值(離散訊號的 fft 結果，第0個值通常不使用。)

    Returns:
        針對每個 row 做fft 的結果。
    """
    _fft = np.fft.fft(node_representation, axis=1)
    res = np.abs(_fft)
    if remove_first:
        res = res[:, 1:]
    return res

def do_otsu_for_avg_fft_spectrum(avg_spectrum):
    """
    回傳 avg_spectrum 的閥值，實作取決於實際算法
    """
    assert avg_spectrum.ndim == 1  # 資料必須是一維

    for _ in avg_spectrum:  # 數值必需在 0~255
        assert 0 < _ < 255  # 數值必需在 0~255

    for_otsu_dtype = np.array(avg_spectrum, dtype=np.uint8)

    ret, _ = cv2.threshold(for_otsu_dtype, 1, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return ret

def get_all_neuron_fft_avg(node_representation):
    """

    Args:
        node_representation:
            2維-的 latent space 的值，
            矩陣高度為 latent neuron 數量，
            矩陣寬度跟時間長度相關，時間越長寬度越長。

    Returns:
        所有神經元輸出，並獨自做fft，加總並平均的 spectrum。
    """
    fft_res = do_latent_representation_fft(node_representation,
                                           remove_first=True)
    # 計算全node平均
    all_node_fft_sum = np.sum(fft_res, axis=0)
    _ = all_node_fft_sum / 2400.0
    assert all_node_fft_sum.ndim == 1  # 因為下方使用 .size 作取值，這邊要小心。
    all_node_fft_avg = _[:all_node_fft_sum.size // 2 + 1]

    # 這個 thres 是拿來畫在 fft 上的，不一定會用到。
    thres = do_otsu_for_avg_fft_spectrum(all_node_fft_avg)

    DRAW_IT = False
    if DRAW_IT:
        print("threshold:", thres)
        plt.plot(all_node_fft_avg)  # float
        # plt.plot(for_otsu_dtype)  # uint8
        plt.axvline(x=thres, color='red')
        plt.show()

    return all_node_fft_avg


def neuron_idx_classify(threshold, every_neuron_fft_peak, delta=0.0):
    high, mid, low = [], [], []
    for idx, peak in enumerate(every_neuron_fft_peak):
        if peak > threshold+delta:
            high.append(idx)
        elif peak < threshold-(delta):
            low.append(idx)
        else:
            mid.append(idx)
    assert len(high) + len(low) + len(mid) == every_neuron_fft_peak.size
    print(f"低於閥值: {len(low)} 個, 高於閥值: {len(high)} 個")
    return high, mid, low

def seperate_latent_representation(node_representation, threshold, delta=0.0):
    """ 
    取得一張 node_representation (高度為 neuron 數量，寬度大小跟 時間有關)
    針對 他的每個node做FFT，並加總平均這些統計圖(frequency domain)
    做 根據閥值 (或者說 position)
    小於閥值的畫成一張 latent matrix :A
    大於閥值的畫成一張 latent matrix :B
    return  (A,B)  兩張大小一樣的 latent matrix
    """
    #
    fft_res = do_latent_representation_fft(node_representation,
                                           remove_first=True)
    fft_res = fft_res[:, :fft_res.shape[1]//2+1]  # 取半

    # 取得每一個 neuron 的 fft pick 數值
    every_neuron_fft_peak = np.array([np.argmax(_) for _ in fft_res])

    # 高於閥值的與低於閥值的 idx 分開。
    low_idx, mid_idx, high_idx = neuron_idx_classify(threshold,
                                                     every_neuron_fft_peak,
                                                     delta=delta)

    # 決定 mid_idx 到底要分為高 or 低，預設視為低。
    mid_idx_is_low = True
    process_mid = False  # mid 是否加到任意一邊

    if process_mid:
        if mid_idx_is_low:
            low_idx += mid_idx
        else:
            high_idx += mid_idx
    else:
        pass  # mid 不會被加到任意一邊

    min_value = np.min(node_representation)  # 基本上是 0

    # 回傳的
    low_latent_image, high_latent_image = None, None

    latent_copy_ = np.copy(node_representation)
    # 先處理 小於 閥值的
    for idx in range(latent_copy_.shape[0]):
        if idx in low_idx:
            latent_copy_[idx] = min_value
    low_latent_image = np.copy(latent_copy_)

    latent_copy_ = np.copy(node_representation)
    # 處理 大於 閥值的
    for idx in range(latent_copy_.shape[0]):
        if idx in high_idx:
            latent_copy_[idx] = min_value
    high_latent_image = np.copy(latent_copy_)

    # 不處理 mid 狀態 就要把他們都關閉
    if not process_mid:
        # 先處理 小於 閥值的
        for idx in range(low_latent_image.shape[0]):
            if idx in mid_idx:
                low_latent_image[idx] = min_value
        # 先處理 大於 閥值的
        for idx in range(high_latent_image.shape[0]):
            if idx in mid_idx:
                high_latent_image[idx] = min_value


    return low_latent_image, high_latent_image

def analysis_latent_space_representation(net, dataloader):
    encoder = net.encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        net.cuda(device)

    latent_matrix = None
    for data in dataloader:
        latent_code = encoder(data.to(device, dtype=torch.float))
        if latent_matrix is None:
            latent_matrix = latent_code
        else:
            latent_matrix = torch.cat((latent_matrix, latent_code), axis=0)

    # 讓 latent neuron 數量 成為 矩陣高度。
    node_representation = latent_matrix.T.detach().cpu().numpy()
    eps_ = np.finfo(float).eps  # 取得系統最小值
    node_representation[node_representation == 0] = eps_  # 取代最小值

    audio_total = 10  # 音源總長度
    audio_take_first = 10  # 擷取前面的 n秒 做分析
    print(f"音源全長 {audio_total} 秒, 採前面 {audio_take_first} 秒 作分析")

    for idx in tqdm(range(len(node_representation))):
        analysis_plot_neuron_representation(node_representation[idx],
                                            str(idx),
                                            audio_total,
                                            audio_take_first
                                            )
    pass


def get_node_representation(net, dataloader):
    encoder = net.encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        net.cuda(device)

    latent_matrix = None
    for data in dataloader:
        latent_code = encoder(data.to(device, dtype=torch.float))
        if latent_matrix is None:
            latent_matrix = latent_code
        else:
            latent_matrix = torch.cat((latent_matrix, latent_code), axis=0)

    # 讓 latent neuron 數量 成為 矩陣高度。
    node_representation = latent_matrix.T.detach().cpu().numpy()
    eps_ = np.finfo(float).eps  # 取得系統最小值
    node_representation[node_representation == 0] = eps_  # 取代最小值

    return node_representation


def get_avg_fft_spectrum(net, dataloader):
    """
    Returns: 經過加權平均後的所有 latent neuron 的 spectrum
    """
    node_representation = get_node_representation(net, dataloader)

    # 分析 latent matrix 的所有 neuron 的 fft 平均值。
    latent_matrix_fft_avg_spectrogram = get_all_neuron_fft_avg(node_representation)

    return latent_matrix_fft_avg_spectrogram

def get_binearlization_latent_matrix(net, dataloader):
    
    # 分析 latent matrix 的所有 neuron 的 fft 平均值。
    latent_matrix_fft_avg_spectrogram = get_avg_fft_spectrum(net, dataloader)
    # 計算 otsu 閥值
    threshold_position = do_otsu_for_avg_fft_spectrum(latent_matrix_fft_avg_spectrogram)
    print("threshold_position:", threshold_position)
    #
    node_representation = get_node_representation(net, dataloader)
    #
    low, high = seperate_latent_representation(node_representation,
                                               threshold_position,
                                               delta=0)

    return low, high


def get_binearlization_latent_matrix_by_fix_threshold(net, dataloader, thres):
    node_representation = get_node_representation(net, dataloader)

    low, high = seperate_latent_representation(node_representation, thres)

    return low, high

def plot_avg_fft(net, dataloader):
    spec = get_avg_fft_spectrum(net, dataloader)
    plt.plot(spec[:50])
    plt.axvline(x=4, color='red')
    plt.show()
    pass

def get_model_input_dataloader(audio_list: List,
                               shuffle: bool,
                               num_workers: int,
                               pin_memory: bool,
                               FFT_dict: Dict,
                               args) -> List[DataLoader]:
    """

    Args:
        audio_list: 音訊位置
        shuffle:  整張FFT 是否要亂序。
        num_workers: windows OS 下請設定 0。
        pin_memory: 電腦記憶體很夠 就設定 True。(增加資料轉移效能)
        FFT_dict (object) : FFT 參數
        args: 全域參數(訓練參數)

    Returns:

    """
    res: List[DataLoader[Any]] = []
    for path in audio_list:
        res.append(hl_dataloader(audio_list,
                                 batch_size=args.batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 FFT_dict=FFT_dict,
                                 args=args)
                   )
        
    return res
