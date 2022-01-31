import numpy as np
from torch.utils.data import DataLoader
from dataset.HLsep_dataloader import hl_dataloader
from typing import List, Dict, Any, AnyStr
import torch
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
from tqdm import tqdm
import cv2


class LatentAnalyzer:
    """
    latent neuron node 訊號分析
    """
    def __init__(self, net, dataloader, audio_length, audio_analysis_used):
        self.net = net
        self.dataloader = dataloader
        self.audio_length = audio_length
        self.audio_analysis_used = audio_analysis_used
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # --------------------- 後續參數
        self.encoder = None
        self.node_representation = None
        self.fft_plot_length = 'all'
        # --------------------- (最後/其他)初始化
        self._finally_init()
        # --------------------- instance attribute assignment
        self._set_node_representation()

    def _finally_init(self):
        if torch.cuda.is_available():
            self.net.cuda(self.device)
        self.encoder = self.net.encoder

    def _set_node_representation(self, replace_zero=True):
        """
        讓 encoder 生成 latent code。
        ---
        @param replace_zero: 是否將 0 取代為最小值(系統依賴)。
        """
        encoder = self.encoder
        dataloader = self.dataloader
        latent_matrix = None
        for data in dataloader:
            latent_code = encoder(data.to(self.device, dtype=torch.float))
            if latent_matrix is None:
                latent_matrix = latent_code
            else:
                latent_matrix = torch.cat((latent_matrix, latent_code), axis=0)

        # 讓 latent neuron 數量 成為 矩陣高度。
        node_representation = latent_matrix.T.detach().cpu().numpy()
        if replace_zero:
            eps_ = np.finfo(float).eps  # 取得系統最小值
            node_representation[node_representation == 0] = eps_  # 取代最小值

        self.node_representation = node_representation

    def plot_all_neuron_fft_representation(self):
        encoder = self.encoder
        net = self.net

        for idx in tqdm(range(len(self.node_representation))):
            self._plot_neuron_fft_representation(self.node_representation[idx],
                                                str(idx))

    def _plot_neuron_fft_representation(self, _1d_tensor, save_name):
        # 因為全長 10秒 作分析太多了故 擷取部分長度即可
        audio_total = self.audio_length  # 音源全長
        audio_take_first = self.audio_analysis_used  # 開頭前 n 秒

        # 計算 擷取占比
        _afp = audio_take_first / audio_total

        # 截斷後長度
        cut_length = int(len(_1d_tensor) * _afp)
        # 取得截斷版本
        _1d_tensor_short = _1d_tensor[0:cut_length]

        ax = plt.subplot(211)
        ax.set_title("latent representation")
        plt.plot(_1d_tensor_short)

        ax = plt.subplot(212)
        #_1d_tensor_short_odd_minus1 = np.copy(_1d_tensor_short)
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
        plt.axvline(x=pick_vline, color='red', linewidth=1, alpha=0.5)
        ax.set_title(f"fft latent representation, p={pick_vline}")

        #
        plt.tight_layout()

        #
        dir_name = './latent_code_fft_analysis'
        os.makedirs(dir_name, exist_ok=True)

        save_name = pjoin(dir_name, save_name)
        plt.savefig(save_name)
        # plt.close()
        # plt.cla()
        plt.clf()

    def get_otsu_threshold(self):
        """
        Returns:
            所有神經元輸出，並獨自做fft，加總並平均的 spectrum並做 otsu 做二分。
        """
        all_node_fft_avg = self._get_all_neuron_fft_avg()

        # 這個 thres 是拿來畫在 fft 上的，不一定會用到。
        thres = self._do_otsu_for_avg_fft_spectrum(all_node_fft_avg)

        return thres

    def _get_all_neuron_fft_avg(self):
        """
        Returns:
            所有神經元輸出，並獨自做fft，加總並平均的 spectrum。
        """
        fft_res = self._do_latent_representation_fft(remove_first=True)
        # 計算全node平均
        all_node_fft_sum = np.sum(fft_res, axis=0)
        _ = all_node_fft_sum / 2400.0
        assert all_node_fft_sum.ndim == 1  # 因為下方使用 .size 作取值，這邊要小心。
        all_node_fft_avg = _[:all_node_fft_sum.size // 2 + 1]

        return all_node_fft_avg

    def _get_plot_neuron_fft_avg(self):
        """
        Returns: 要拿來繪製的 fft signal，回傳不特定長度
        (內容為所有神經元輸出，並獨自做fft，加總並平均的 spectrum。)
        """
        fft_res = self._do_latent_representation_fft(remove_first=True)
        # 計算全node平均
        all_node_fft_sum = np.sum(fft_res, axis=0)
        _ = all_node_fft_sum / 2400.0
        assert all_node_fft_sum.ndim == 1  # 因為下方使用 .size 作取值，這邊要小心。
        all_node_fft_avg = _[:all_node_fft_sum.size // 2 + 1]

        if isinstance(self.fft_plot_length, str):
            assert self.fft_plot_length.lower() == 'all'
        else:
            assert 0 < self.fft_plot_length < len(all_node_fft_avg)
            all_node_fft_avg = all_node_fft_avg[0:self.fft_plot_length]

        return all_node_fft_avg

    def plot_avg_fft(self, plot_otsu=True, plot_axvline=0.0):

        # 這個 thres 是拿來畫在 fft 上的，不一定會用到。
        thres = self.get_otsu_threshold()

        print("otsu threshold:", thres)
        plt.title(f"Latent layer neuron fft avg (otsu threshold: {thres})")
        plt.plot(self._get_plot_neuron_fft_avg())  # float
        plt.xlim(1, len(self._get_plot_neuron_fft_avg()))
        # plt.plot(for_otsu_dtype)  # uint8
        if plot_otsu:
            plt.axvline(x=thres, color='green', alpha=0.5)
        if plot_otsu and plot_axvline != 0.0:
            print("[warn]: 請選擇一個 垂直線 繪製!!")
        if plot_axvline != 0.0:
            plt.axvline(x=plot_axvline, color='green', alpha=0.5)
            plt.axvline(x=thres, color='red', alpha=0.2)
        plt.show()

    def _do_otsu_for_avg_fft_spectrum(self, avg_spectrum):
        """
        回傳 avg_spectrum 的閥值，實作取決於實際算法
        """
        assert avg_spectrum.ndim == 1  # 資料必須是一維

        for _ in avg_spectrum:
            assert 0 < _ < 255  # 數值必需在 0~255，因為 cv2 的算法只支援到 uint8

        for_otsu_dtype = np.array(avg_spectrum, dtype=np.uint8)

        ret, _ = cv2.threshold(for_otsu_dtype, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return ret


    def _do_latent_representation_fft(self, remove_first):
        """
            Returns:
                針對每個 row 做fft 的結果。
        """
        _fft = np.fft.fft(self.node_representation, axis=1)
        res = np.abs(_fft)
        if remove_first:
            res = res[:, 1:]
        return res

    def _neuron_idx_classify(self, fix_thres_threshold=-1, delta=0.0):
        if fix_thres_threshold == -1:
            threshold = self.get_otsu_threshold()
        else:
            print(f"[warn]: Used the fix threshold = {fix_thres_threshold}.")
            threshold = fix_thres_threshold
        every_neuron_fft_peak = self._get_every_node_fft_peak()
        high, low = [], []
        for idx, peak in enumerate(every_neuron_fft_peak):
            if peak > threshold + delta:
                high.append(idx)
            elif peak <= threshold - delta:
                low.append(idx)
            else:
                pass

        print(f"低於閥值: {len(low)} 個, 高於閥值: {len(high)} 個")
        return high, low

    def _get_every_node_fft_peak(self):
        """
        取得一張 node_representation (高度為 neuron 數量，寬度大小跟 時間有關)
        針對 他的每個node做FFT，並加總平均這些統計圖(frequency domain)
        做 根據閥值 (或者說 position)
        小於閥值的畫成一張 latent matrix :A
        大於閥值的畫成一張 latent matrix :B
        return  (A,B)  兩張大小一樣的 latent matrix
        """
        #
        fft_res = self._do_latent_representation_fft(remove_first=True)
        fft_res = fft_res[:, :fft_res.shape[1] // 2 + 1]  # 取半

        # 取得每一個 neuron 的 fft pick 數值
        every_neuron_fft_peak = np.array([np.argmax(_) for _ in fft_res])

        return every_neuron_fft_peak

    def get_binearlization_latent_matrix_by_fix_threshold(self, fix_thres,
                                                          delta=0.0, plot=False,
                                                          plot_otsu=False):
        if fix_thres-delta <= 0:
            print(f"[Error]: 左邊的 threshold<0。 (threshold={fix_thres-delta})")
        low, high = self._get_seperate_latent_representation_by_fix_threshold(
            fix_thres,
            delta=delta)

        if plot:
            sp = self._get_plot_neuron_fft_avg()
            plt.plot(sp)
            plt.axvline(x=fix_thres, color='red', alpha=0.5)
            if delta != 0.0:
                plt.axvspan(fix_thres-delta, fix_thres+delta, facecolor='red', alpha=0.1)
            if plot_otsu:
                otsu = self.get_otsu_threshold()
                plt.axvline(x=otsu, color='green', alpha=0.5)
            plt.title(f"Latent layer neuron fft avg(fix threshold={fix_thres}, otsu={otsu})")
            plt.show()

        return low, high

    def get_binearlization_latent_matrix(self):
        low, high = self._get_seperate_latent_representation()

        return low, high

    def _get_seperate_latent_representation(self, delta=0.0):
        """
        取得一張 node_representation (高度為 neuron 數量，寬度大小跟 時間有關)
        針對 他的每個node做FFT，並加總平均這些統計圖(frequency domain)
        做 根據閥值 (或者說 position)
        小於閥值的畫成一張 latent matrix :A
        大於閥值的畫成一張 latent matrix :B
        return  (A,B)  兩張大小一樣的 latent matrix
        """
        if delta != 0.0:
            print(f"[warn]: delta={delta}")

        # 高於閥值的與低於閥值的 idx 分開。
        low_idx, high_idx = self._neuron_idx_classify(delta=delta)

        min_value = np.min(self.node_representation)  # 基本上是 0

        latent_copy_ = np.copy(self.node_representation)
        # 先處理 小於 閥值的
        for idx in range(latent_copy_.shape[0]):
            if idx in low_idx:
                latent_copy_[idx] = min_value
        low_latent_image = np.copy(latent_copy_)

        latent_copy_ = np.copy(self.node_representation)
        # 處理 大於 閥值的
        for idx in range(latent_copy_.shape[0]):
            if idx in high_idx:
                latent_copy_[idx] = min_value
        high_latent_image = np.copy(latent_copy_)

        return low_latent_image, high_latent_image


    def _get_seperate_latent_representation_by_fix_threshold(self, threshold, delta=0.0):
        """
        取得一張 node_representation (高度為 neuron 數量，寬度大小跟 時間有關)
        針對 他的每個node做FFT，並加總平均這些統計圖(frequency domain)
        根據 threshold (或者說 position)
        小於閥值的畫成一張 latent matrix :A
        大於閥值的畫成一張 latent matrix :B
        return  (A,B)  兩張大小一樣的 latent matrix
        """
        if delta != 0.0:
            print(f"[warn]: delta={delta}")

        # 高於閥值的與低於閥值的 idx 分開。
        low_idx, high_idx = self._neuron_idx_classify(delta=delta,
                                                     fix_thres_threshold=threshold)

        min_value = np.min(self.node_representation)  # 基本上是 0

        # 回傳結果
        low_latent_image, high_latent_image = None, None

        latent_copy_ = np.copy(self.node_representation)
        # 先處理 小於 閥值的
        for idx in range(latent_copy_.shape[0]):
            if idx in low_idx:
                latent_copy_[idx] = min_value
        low_latent_image = np.copy(latent_copy_)

        latent_copy_ = np.copy(self.node_representation)
        # 處理 大於 閥值的
        for idx in range(latent_copy_.shape[0]):
            if idx in high_idx:
                latent_copy_[idx] = min_value
        high_latent_image = np.copy(latent_copy_)

        return low_latent_image, high_latent_image

