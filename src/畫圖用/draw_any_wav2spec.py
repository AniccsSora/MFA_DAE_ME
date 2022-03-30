import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import scipy.io.wavfile as wav
from adobe_colorbar import Adobe_color_maker


mpl.rcParams['font.size'] = 20
#mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'large'


class WavSpecDrawer:
    def __init__(self):
        self.FFT_dict = {
            'sr': 8000,
            'frequency_bins': [0, 300],
            'FFTSize': 2048,  # 2048
            'Hop_length': 128,  # 128
            'Win_length': 2048,  # 2048
        }

    def draw_spec(self, path):
        spec, y = self._y2lps(path)
        self._draw_spec(spec, len(y))

    def _y2lps(self, wav_path):
        """
        Args:
            wav_path: wav 路徑

        Returns:
            Sxx: np.array: 2維,
             y : 讀近來訊號
        """
        sr, y = wav.read(wav_path)
        if y.dtype == np.int16:
            y = y / 32767.
        else:
            raise NotImplementedError(f"Not support WAV format. : {y.dtype}")

        # 轉 絕對值
        epsilon = np.finfo(float).eps
        D = librosa.stft(y, n_fft=self.FFT_dict['FFTSize'],
                         hop_length=self.FFT_dict['Hop_length'],
                         win_length=self.FFT_dict['Win_length'],
                         window=scipy.signal.hamming,
                         center=False)
        D = D + epsilon

        Sxx = np.log10(abs(D) ** 2)

        return Sxx, y

    def _draw_spec(self, spec, origin_y_len):
        """
        非常 hardcode 的寫法 :)
        """
        max_sec = origin_y_len / self.FFT_dict['sr']
        _sec_step = 2  # (sec) 每個 tick 差幾秒
        each_tick_sec = self.FFT_dict['Hop_length'] / self.FFT_dict['sr']
        # 計算跳步
        _tick_step = _sec_step / each_tick_sec

        # arange 生成 tick
        xs_tick = np.arange(0, spec.shape[1], _tick_step)
        sec_tick = np.arange(0, max_sec, _sec_step, dtype=np.int64)
        # print("xs_tick:", xs_tick)
        # print("sec_tick:", sec_tick)  # label

        # y 的 hardcode 軸
        yy_ticks = [0, 512, 1025]
        yy_labels = [1024, 512, 0]

        adobe_cb = Adobe_color_maker()
        my_cb = adobe_cb.get_colorbar()
        adobe_cb.get_colorbar()

        fig = plt.figure(figsize=(12, 8), dpi=300)
        ax = plt.gca()
        ax.spines[['top', 'right']].set_visible(False)
        plt.imshow(spec[::-1], cmap=my_cb)

        ax.set_xticks(ticks=xs_tick, labels=sec_tick)
        ax.set_xlabel('Sec', fontsize=26)

        ax.set_yticks(ticks=yy_ticks, labels=yy_labels)
        ax.set_ylabel('Frequency (Hz)', fontsize=26)

        # Normalizer
        norm = mpl.colors.Normalize(vmin=np.min(spec[::-1]),
                                    vmax=np.max(spec[::-1]))

        # creating ScalarMappable
        sm = plt.cm.ScalarMappable(cmap=my_cb, norm=norm)
        sm.set_array([])

        # plt.colorbar(sm, ticks=np.linspace(0, 2, 2))
        plt.colorbar(sm)
        #plt.savefig("好看頻譜圖.png")
        #plt.show()


if __name__ == "__main__":

    hi = WavSpecDrawer()

    a1 = r"D:\Git\MFA_DAE_ME\src\log\DAE_C_2022_0330_2339_01\test_my_source1\121_1b1_Tc_sc_Meditron.wav"
    a2 = r"D:\Git\MFA_DAE_ME\src\log\DAE_C_2022_0330_2339_01\test_my_source2\121_1b1_Tc_sc_Meditron.wav"

    hi.draw_spec(a1)
    plt.savefig("1.png")

    hi.draw_spec(a2)
    plt.savefig("2.png")



