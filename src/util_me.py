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
    ax.set_title("fft latent representation")
    _ = np.abs(np.fft.fft(_1d_tensor_short))  # 做一維傅立葉變換
    _ = np.log10(_)  # 取 log
    plt.plot(_[:len(_)//2+1])

    ax = plt.subplot(224)
    ax.set_title("fft latent representation (log2)")
    _ = np.abs(np.fft.fft(_1d_tensor_short_log2))
    _ = np.log10(_)
    plt.plot(_[:len(_)//2+1])

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
    audio_take_first = 2.2  # 擷取前面的 n秒 做分析
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