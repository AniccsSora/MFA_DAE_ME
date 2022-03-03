import numpy as np
from torch.utils.data import DataLoader
from dataset.HLsep_dataloader import hl_dataloader
from typing import List, Dict, Any, AnyStr
from scipy.signal import peak_prominences, find_peaks, find_peaks_cwt
import scipy
import torch
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
from tqdm import tqdm
import cv2
from matplotlib.scale import FuncScale

class LatentAnalyzer:
    """
    latent neuron node 訊號分析
    """
    def __init__(self, net, dataloader, audio_length, audio_analysis_used):
        self.net = net
        self.dataloader = dataloader
        self.audio_length = audio_length  # 秒數
        self.audio_analysis_used = audio_analysis_used
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # --------------------- 後續參數
        self.encoder = None
        self.node_representation = None
        self.fft_plot_length = 'all'
        self._latent_n_number = 2400  # 模型的 latent neuron 數量
        self._sample_rate = 8000
        # ---------------------  static 表示 特殊靜態變數，如要安全取用，請呼叫他的更新
        self.__static_all_neuron_fft_avg_ = None
        self.__rfft_freq = None
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

    def plot_all_neuron_fft_representation(self, limit=0):
        encoder = self.encoder
        net = self.net
        if limit == 0:
            limit = 99999999
        for idx in tqdm(range(len(self.node_representation))):
            self._plot_neuron_fft_representation(self.node_representation[idx],
                                                str(idx))
            if idx >= limit:
                print(f"end of limit:{limit}")
                break

    def plot_all_fft_latent_neuron_peaks(self, limit=0):
        encoder = self.encoder
        net = self.net
        if limit == 0:
            limit = 999999999
        for idx in tqdm(range(len(self.node_representation))):
            signal = self.node_representation[idx]
            #
            _fft = np.fft.rfft(signal)[1:]  # 輸出結果第一個不取
            res_no_smooth = np.abs(_fft)  # fft 結果
            res_freq = np.fft.rfftfreq(n=signal.size, d=1. / self._sample_rate)[1:]
            res = self._smooth(res_no_smooth, window_len=11)

            peaks_idx, _ = find_peaks(res,
                                      distance=5,  # peaks 間最小距離
                                      height=0)  # height= ([最小高度], [最大高度])
            prominences = peak_prominences(res, peaks_idx)[0]  # 計算突出值
            contour_heights = res[peaks_idx] - prominences  # 突出線條
            plt.title(f"Filter Peaks show: No.{idx}")
            plt.plot(res_freq, res)
            plt.plot(res_freq, res_no_smooth, alpha=.5)
            # plt.vlines(x=peaks_idx, ymin=contour_heights, ymax=signal[peaks_idx])
            new_peaks_idx, _ = self._calc_most_height_peaks((peaks_idx, prominences),
                                                            percent=.9)
            # plt.plot(peaks_idx, res[peaks_idx], ".", alpha=.3)  # 全部 peaks 畫出來
            plt.plot(res_freq[new_peaks_idx], res[new_peaks_idx], "x", color='red')  # 挑最大的畫出來
            # 繪製對用於原始訊號位置的 peaks
            #plt.plot(res_freq[new_peaks_idx], res_no_smooth[new_peaks_idx], "o", color='red')
            #plt.show()
            dir_name = './latent_code_fft_peaks (smoooth)'
            os.makedirs(dir_name, exist_ok=True)
            save_name = pjoin(dir_name, str(idx)+'.png')
            plt.savefig(save_name)
            plt.clf()

            if idx >= limit:
                print(f"end of limit:{limit}")
                break


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
        # _ = np.abs(np.fft.rfft(_1d_tensor_short_odd_minus1))  # 做一維傅立葉變換 奇數idx 值*-1

        _ = np.abs(np.fft.rfft(_1d_tensor_short)[1:])  # 做 rfft，去除第一個元素
        assert _.ndim == 1
        _freq = np.fft.rfftfreq(_.size, d=1./self._sample_rate)
        # _ = np.log10(_)  # ------------------------------------  取 log
        draw_val = _
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
            所有神經元輸出，並獨自做 rfft，加總並平均的 spectrum並做 otsu 做二分。
        """
        all_node_fft_avg = self._get_all_neuron_fft_avg()

        # 這個 thres 是拿來畫在 fft 上的，不一定會用到。
        thres = self._do_otsu_for_avg_fft_spectrum(all_node_fft_avg)

        return thres

    def get_otsu_threshold_freq_ver(self):
        """
        Returns:
            get_otsu_threshold 的 freq 刻度版本
        """
        thres = int(self.get_otsu_threshold())

        return self._get_rfft_freq()[thres+1]

    def _get_all_neuron_fft_avg(self):
        """
        Returns:
            所有神經元輸出，並獨自做 real-fft，加總並平均的 spectrum。
        """
        fft_res, freq = self._do_latent_representation_rfft(remove_first=True)
        # 計算全node平均
        all_node_fft_sum = np.sum(fft_res, axis=0)

        res = all_node_fft_sum / self._latent_n_number

        return res

    def _get_plot_neuron_fft_avg(self):
        """
        Returns: 要拿來繪製的 fft signal，回傳不特定長度
        (內容為所有神經元輸出，並獨自做fft，加總並平均的 spectrum。)
        """
        fft_res, freq = self._do_latent_representation_rfft(remove_first=True)
        # 計算全node平均
        all_node_fft_sum = np.sum(fft_res, axis=0)
        _ = all_node_fft_sum / self._latent_n_number

        all_node_fft_avg = _

        if isinstance(self.fft_plot_length, str):
            assert self.fft_plot_length.lower() == 'all'
        else:
            assert 0 < self.fft_plot_length < len(all_node_fft_avg)
            all_node_fft_avg = all_node_fft_avg[0:self.fft_plot_length]

        return all_node_fft_avg, freq

    def plot_avg_fft(self, plot_otsu=True, plot_axvline=0.0, freq_tick=True, x_log_scale=True, smooth=True):
        plt.clf()
        # 這個 thres 是拿來畫在 fft 上的，不一定會用到。
        if freq_tick:
            thres = int(self.get_otsu_threshold_freq_ver())
        else:
            thres = self.get_otsu_threshold()

        print(f"otsu threshold (freq_tick={freq_tick}):", thres)

        if x_log_scale:
            ax = plt.gca()
            ax.set_xscale('log')

        plt.title(f"Latent layer neuron fft avg (otsu threshold: {thres})")
        _draw_target, freq = self._get_plot_neuron_fft_avg()
        _orignal_draw_target = np.copy(_draw_target)
        if smooth:
            #_draw_target = self._smooth_signal(_draw_target)
            wl = 11
            _draw_target = self._smooth(_draw_target, window_len=wl)
            #_shift = (wl-1)//2
            #_draw_target = _draw_target[_shift:len(_draw_target)-_shift]
        # 劃出 peak
        self._plot_signal_peak(default_sig=_draw_target)

        if freq_tick:
            ax = plt.gca()
            ax.set_title("freq. x-tick, FFT (smooth v.s. original)")
            plt.plot(freq, _draw_target)
            plt.plot(freq, _orignal_draw_target, color='C0', alpha=0.5)
        else:
            plt.plot(_draw_target)
            plt.plot(_orignal_draw_target, color='C0', alpha=0.5)
        if plot_otsu:
            plt.axvline(x=thres, color='green', alpha=0.5)
            plt.axvline(x=0, color='red', alpha=0.5, linewidth=1)
        if plot_otsu and plot_axvline != 0.0:
            print("[warn]: 請選擇一個 垂直線 繪製!!")
        if plot_axvline != 0.0:
            plt.axvline(x=plot_axvline, color='green', alpha=0.5)
            plt.axvline(x=thres, color='red', alpha=0.2)


    def _smooth_signal(self, sig):
        win = scipy.signal.windows.hann(10)
        filtered = scipy.signal.convolve(sig, win, mode='same') / sum(win)
        return filtered

    def _do_otsu_for_avg_fft_spectrum(self, avg_spectrum):
        """
        回傳 avg_spectrum 的閥值，實作取決於實際算法
        """
        assert avg_spectrum.ndim == 1  # 資料必須是一維

        # 測試這樣搞 O不OK
        avg_spectrum = np.clip(avg_spectrum, 0, 255)

        for _ in avg_spectrum:
            assert 0 <= _ <= 255  # 數值必需在 0~255，因為 cv2 的算法只支援到 uint8
            if (0 < _ < 255) is False:
                print("max:", np.max(avg_spectrum))
                print("min:", np.min(avg_spectrum))

        for_otsu_dtype = np.array(avg_spectrum, dtype=np.uint8)

        ret, _ = cv2.threshold(for_otsu_dtype, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return ret


    def _do_latent_representation_rfft(self, remove_first):
        """
            Returns:
                針對每個 row 做 real-fft 的結果。
        """
        _fft = np.fft.rfft(self.node_representation, axis=1)
        res = np.abs(_fft)
        res_freq = np.fft.rfftfreq(n=self.node_representation[0].size, d=1. / self._sample_rate)
        self.__rfft_freq = res_freq
        if remove_first:
            res = res[:, 1:]
            res_freq = res_freq[1:]

        return res, res_freq

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
        fft_res, freq = self._do_latent_representation_rfft(remove_first=True)
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
            sp, freq = self._get_plot_neuron_fft_avg()
            plt.plot(freq, sp)
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

    def _plot_signal_peak(self, show_all_peaks=True, default_sig=None):
        """
        Returns: 分析 neuron 的 訊號突出點，並返回
        """
        # peak_prominences()  # 計算峰值的突出度
        # find_peaks()  # 尋找波峰
        if default_sig is None:
            signal = self._get_static_all_neuron_fft_avg_(update=True)
        else:
            signal = default_sig

        # 這會找到很多波峰，全部凸的都會出來
        # peaks_idx = find_peaks_cwt(signal, widths=1)  # widths 感興趣的峰值寬度
        peaks_idx, _ = find_peaks(signal,
                                  distance=5,  # peaks 間最小距離
                                  height=0)  # height= ([最小高度], [最大高度])
        prominences = peak_prominences(signal, peaks_idx)[0]  # 計算突出值
        contour_heights = signal[peaks_idx] - prominences  # 突出線條
        plt.clf()
        plt.title("Filter Peaks show")
        plt.plot(signal)
        # plt.vlines(x=peaks_idx, ymin=contour_heights, ymax=signal[peaks_idx])
        new_peaks_idx, _ = self._calc_most_height_peaks((peaks_idx, prominences))
        plt.plot(peaks_idx, signal[peaks_idx], ".", alpha=.5)   # 全部畫出來
        plt.plot(new_peaks_idx, signal[new_peaks_idx], "x", color='red')  # 挑最大的畫出來


        def draw_all_peaks():
            _d_peaks_idx, _ = find_peaks(signal, height=0)
            plt.title("All peaks show")
            plt.plot(signal)
            plt.plot(_d_peaks_idx, signal[_d_peaks_idx], ".")
            #plt.show()

        if show_all_peaks:
            draw_all_peaks()


    def _calc_most_height_peaks(self, peaks_pair: tuple, percent=95):
        """
        Args:
            peaks_pair: 訊號 idx，該 idx 的峰值
            percent: 當作 PR 值 (取峰值高的)。
        Returns:
            返回所需要的 peaks_idx(峰值索引，根據原來的訊號總長度), peaks_prominence(峰值)
        """
        assert len(peaks_pair) == 2
        assert peaks_pair[0].shape == peaks_pair[1].shape
        idx, promi = peaks_pair
        mask = promi > np.percentile(promi, percent)
        new_idx, new_promi = idx[mask], promi[mask]

        return new_idx, new_promi

    def _get_static_all_neuron_fft_avg_(self, update: bool):
        if self.__static_all_neuron_fft_avg_ is None:
            self._update_static_all_neuron_fft_avg_()
        elif update:
            self._update_static_all_neuron_fft_avg_()

        return self.__static_all_neuron_fft_avg_

    def _update_static_all_neuron_fft_avg_(self):
        self.__static_all_neuron_fft_avg_ = self._get_all_neuron_fft_avg()

    def _get_rfft_freq(self):
        """
        取得 real-fft 的 freq. tick
        """
        if self.__rfft_freq is None:
            self._do_latent_representation_rfft()
        return self.__rfft_freq

    def _smooth(self, x, window_len=3, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            if window == 'hanning':
                w = np.hanning(window_len)
            else:
                w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        _shift = (window_len - 1) // 2
        y = y[_shift:len(y) - _shift]

        return y


def mel(*a):
    a = np.array(a)
    assert a.ndim == 1
    return 2410*np.log10(1+a/625)


if __name__ == "__main__":
    x = np.arange(1, 4000)
    y = np.log10(x**2)

    ax = plt.gca()
    yo = FuncScale(ax, mel)
    ax.set_xscale(yo)
    plt.plot(x, y)
    plt.show()
