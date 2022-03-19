import sys

sys.path.append('../')
from torch.utils.data import DataLoader, Dataset
from utils.signalprocess import wav2lps, lps2wav, wav_read
import utils.fake_args as fake_args
import numpy as np
import scipy.io.wavfile as wav


class __HL_multiple_dataset(Dataset):

    def __init__(self, data_path_list, FFT_dict, args, add_slient_seq):

        self.data_path_list = data_path_list
        self.FFT_dict = FFT_dict
        self.args = args
        for idx, filepath in enumerate(self.data_path_list):
            if args.data_feature == "lps":
                spec, phase, mean, std = wav2lps(filepath, self.FFT_dict['FFTSize'], self.FFT_dict['Hop_length'],
                                                 self.FFT_dict['Win_length'], self.FFT_dict['normalize'])
                if args.model_type == "DAE_C":
                    if idx == 0:
                        self.samples = np.reshape((spec.T), (-1, 1, 1, int(self.FFT_dict['FFTSize'] / 2 + 1)))[:, :, :,
                                       self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
                    else:
                        next_ = np.reshape((spec.T), (-1, 1, 1, int(self.FFT_dict['FFTSize'] / 2 + 1)))[:, :, :,
                        self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
                        if add_slient_seq:
                            add_length = 10
                            fill_shape = (add_length, self.samples.shape[-3], self.samples.shape[-2], self.samples.shape[-1])
                            fill_val = np.min(self.samples)
                            _ = np.full(fill_shape, fill_val, dtype=self.samples.dtype)
                            self.samples = np.append(self.samples, _, axis=0)

                        self.samples = np.append(self.samples, next_, axis=0)
                else:
                    self.samples = spec.T[:, self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]

            else:
                raise NotImplementedError("這邊沒有實作 因為實驗只跑lps.")
                y = wav_read(filepath)
                self.samples = np.reshape(y, (-1, 1, 1, y.shape[0]))

    def __getitem__(self, index):

        return self.samples[index]

    def __len__(self):

        return len(self.samples)


def hl_multiple_dataloader(data_path_list, args, FFT_dict,
                           shuffle=False, num_workers=0,
                           pin_memory=True, add_slient_seq=False):

    hl_dataset = __HL_multiple_dataset(data_path_list, FFT_dict, args, add_slient_seq=add_slient_seq)
    hl_dataloader = DataLoader(hl_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True, drop_last=False, )

    return hl_dataloader




# Default fourier transform parameters
FFT_dict = {
    'sr': 8000,
    'frequency_bins': [0, 300],
    'FFTSize': 2048,  # 2048
    'Hop_length': 128,  # 128
    'Win_length': 2048,  # 2048
    'normalize': True,
}

if __name__ == "__main__":
    datalist = ["./4_1.wav",
                "./4_1_5sec.wav"]
    args = fake_args.get_args()

    my_dataloader = hl_multiple_dataloader(datalist, args=args,
                                           FFT_dict=FFT_dict, add_slient_seq=False)

    pass
