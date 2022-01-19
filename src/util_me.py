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

    _1d_tensor_log2 = np.log2(_1d_tensor)

    # 截斷後長度
    cut_length = int(len(_1d_tensor) * _afp)
    # 取得截斷版本
    _1d_tensor_short = _1d_tensor[0:cut_length]
    _1d_tensor_short_log2 = _1d_tensor_log2[0:cut_length]

    ax = plt.subplot(221)
    ax.set_title("latent representation")
    plt.plot(_1d_tensor_short)
    ax = plt.subplot(223)
    ax.set_title("latent representation (log2)")
    plt.plot(_1d_tensor_short_log2)

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
    _freq = np.fft.fftfreq(_.size, d=8000)
    # _ = np.log10(_)  # ------------------------------------  取 log
    draw_val = _[1:len(_) // 2 + 1]  # 不使用 fft 第 0 個的數值
    plt.plot(draw_val)
    pick_vline = np.argmax(draw_val)
    plt.axvline(x=pick_vline, color='red', linewidth=1)
    ax.set_title(f"fft latent representation, p={pick_vline}")

    ax = plt.subplot(224)
    ax.set_title("fft latent representation (log2)")
    _ = np.abs(np.fft.fft(_1d_tensor_short_log2))
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
    _fft = np.fft.fft(node_representation, axis=1)
    res = np.abs(_fft)
    if remove_first:
        res = res[:, 1:]
    return res

def do_otsu_for_avg_fft_spectrum(avg_spectrum):
    """
    回傳閥值
    """
    for _ in avg_spectrum:  # 數值必需在 0~255
        assert 0 < _ < 255  # 數值必需在 0~255

    for_otsu_dtype = np.array(avg_spectrum, dtype=np.uint8)

    ret, _ = cv2.threshold(for_otsu_dtype, 1, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return ret

def analysis_all_neuron_fft_avg(node_representation):
    fft_res = do_latent_representation_fft(node_representation,
                                           remove_first=True)
    # 計算全node平均
    all_node_fft_sum = np.sum(fft_res, axis=0)
    _ = all_node_fft_sum / 2400.0
    all_node_fft_avg = _[:all_node_fft_sum.size // 2 + 1]

    thres = do_otsu_for_avg_fft_spectrum(all_node_fft_avg)
    print("threshold:", thres)
    plt.plot(all_node_fft_avg)  # float
    # plt.plot(for_otsu_dtype)  # uint8
    plt.axvline(x=thres, color='red')
    plt.show()
    pass


def otsu_analysis_for_latent_representation():
    """ 
    取得一張 latent matrix
    針對 他的每個node做FFT，並加總平均這些統計圖(frequcy domain)
    做 otsu 並得到閥值
    小於閥值的畫成一張 latent matrix :A
    大於閥值的畫成一張 latent matrix :B
    retrun A,B
    """

    pass

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

    # prof. 分析
    analysis_all_neuron_fft_avg(node_representation)

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
