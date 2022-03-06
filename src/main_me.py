# -*- coding: utf-8 -*-
import argparse
import time
from utils import misc
import torch
from torch.autograd import Variable
from datetime import datetime
import numpy as np
from model import DAE_C, DAE_F
import train
from source_separation import MFA
from torch.utils.data import DataLoader
from dataset.HLsep_dataloader import hl_dataloader, val_dataloader
import scipy.io.wavfile as wav
import os
from os.path import join as pjoin
import tester
import logging
import util_me as me
from typing import List, Any
from LatentAnalyzer import LatentAnalyzer
import matplotlib.pyplot as plt
import matplotlib

logging.getLogger(__name__)

# parser#
parser = argparse.ArgumentParser(description='PyTorch Source Separation')
parser.add_argument('--model_type', type=str, default='DAE_C', help='model type', choices=['DAE_C', 'DAE_F'])
parser.add_argument('--data_feature', type=str, default='lps', help='lps or wavform')
parser.add_argument('--pretrained', default=False, help='load pretrained model or not')
parser.add_argument('--pretrained_path', type=str, default=None, help='pretrained_model path')
parser.add_argument('--trainOrtest', type=str, default="train", help='status of training')
# training hyperparameters
parser.add_argument('--optim', type=str, default="Adam", help='optimizer for training', choices=['RMSprop', 'SGD', 'Adam'])
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for training (default: 1e-3)')
parser.add_argument('--CosineAnnealingWarmRestarts', type=bool, default=True, help='optimizer scheduler for training')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')

parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
# MFA hyperparameters
parser.add_argument('--source_num', type=int, default=3, help='number of separated sources')
parser.add_argument('--clustering_alg', type=str, default='NMF', choices=['NMF', 'K_MEANS'], help='clustering algorithm for embedding space')
parser.add_argument('--wienner_mask', type=bool, default=True, help='wienner time-frequency mask for output')
#
parser.add_argument('--fix_thres', type=int, default=-1)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
#  misc.logger.init(args.logdir, 'train_log_')  # 拒用
#  logger = misc.logger.info  #  == print()
misc.ensure_dir(args.logdir)
#  取時戳
current_time = datetime.now().strftime('%Y_%m%d_%H%M_%S')
#  拚 log 資料夾名
args.logdir = args.logdir + str(args.model_type) + "_" + str(current_time)
os.makedirs(args.logdir)
#  log 位置與檔案名
log_full_path = pjoin(args.logdir, f'log_{current_time}.txt')
logging.basicConfig(filename=log_full_path, level=logging.INFO, force=True, filemode='w')
logger = logging.info
starttime = time.time()


logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))
logger("========================================")

"""
if args.seed is not None:
    random.seed(args.seed)
    cudnn.deterministic=None
    ngpus_per_node = torch.cuda.device_count()
"""
# build model
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
logger('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
per_save_epoch = 30
t_begin = time.time()
grad_scale = args.grad_scale

# Default model dictionary
DAE_C_dict = {
        "frequency_bins": [0, 300],
        # "encoder": [32, 16, 8],
        # "decoder": [8, 16, 32, 1],
        "encoder": [32, 16, 8],
        "decoder":  [8, 16, 32, 1],
        "encoder_filter": [[1, 3], [1, 3], [1, 3]],
        "decoder_filter": [[1, 3], [1, 3], [1, 3],  [1, 1]],
        "encoder_act": "relu",
        "decoder_act": "relu",
        "dense": [],
        }
# 防呆
assert len(DAE_C_dict['encoder']) == len(DAE_C_dict['encoder_filter'])
assert len(DAE_C_dict['decoder']) == len(DAE_C_dict['decoder_filter'])

DAE_F_dict = {
        "frequency_bins": [0, 300],
        "encoder": [1024, 512, 256, 128],
        "decoder": [256, 512, 1024, 1025],
        "encoder_act": "relu",
        "decoder_act": "relu",
        }

Model = {
    'DAE_C': DAE_C.autoencoder,
    'DAE_F': DAE_F.autoencoder,
}

model_dict = {
    'DAE_C': DAE_C_dict,
    'DAE_F': DAE_F_dict
}


# Default fourier transform parameters
FFT_dict = {
    'sr': 8000,
    'frequency_bins': [0, 300],
    'FFTSize': 2048,  # 2048
    'Hop_length': 128,  # 128
    'Win_length': 2048,  # 2048
    'normalize': True,
}
# declare model object
net = Model[args.model_type](model_dict=model_dict[args.model_type], args=args, logger=logger).cuda()

# torch setting random seed.
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    net.cuda()

matplotlib.rcParams['figure.figsize'] = [8, 6]
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size'] = 14
#plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['figure.titlesize'] = 'large'

if __name__ == "__main__":

    # data loader
    test_filelist = ["./dataset/4_1.wav"]
    #test_filelist = ["./dataset/4_1_5sec.wav"]
    #test_filelist = ["./dataset/senpai_data/heart_lung_sam2/mix/training_noise_呼吸/0dB/4_1.wav"]
    #test_filelist = ["./dataset/senpai_data/heart_lung_sam2/mix/training_noisy_心肺/6dB/3_0.wav"]

    test_filename = test_filelist[0].split('/')[-1].split('.')[0]  # get pure-filename
    outdir = "{}/test_".format(args.logdir)
    # train_loader = hl_dataloader(test_filelist,
    #                              batch_size=args.batch_size,
    #                              shuffle=True,
    #                              num_workers=0,
    #                              pin_memory=False,
    #                              FFT_dict=FFT_dict,
    #                              args=args)

    train_loader_list: List[DataLoader[Any]]
    train_loader_list = me.get_model_input_dataloader(test_filelist,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=False,
                                                      FFT_dict=FFT_dict,
                                                      args=args)
    #
    # train
    net = train.train(train_loader_list[0], net, args, logger)
    #net.load_state_dict(torch.load(r"./log/DAE_C/latest.pt"))

    # 全新物件，全新感受
    LA = LatentAnalyzer(net, train_loader_list[0],
                        audio_length=10, audio_analysis_used=10,
                        args=args)

    # 這個會繪製 每個 neuron 的值與 fft 輸出。
    # me.analysis_latent_space_representation(net, train_loader_list[0])
    #LA.plot_all_fft_latent_neuron_peaks(limit=100)
    #LA.plot_all_neuron_fft_representation(limit=500)

    # 繪製 加權平均 fft
    LA.fft_plot_length = 'All'  # 或者使用 'All'
    # LA.plot_avg_fft(plot_otsu=True, plot_axvline=0, freq_tick=False)
    LA.plot_claen_avg_fft()
    LA.plot_avg_fft(plot_otsu=True, plot_axvline=0, freq_tick=True,
                    x_log_scale=False, smooth=False, save_name="avg_spectrum (otsu)")
    LA.plot_avg_fft(plot_otsu=False, plot_axvline=0, freq_tick=True,
                    x_log_scale=False, smooth=False, save_name="avg_spectrum (no-otsu)")


    # 分析 波
    LA._plot_signal_peak()
    plt.show()

    # Source Separation by MFA analysis.
    mfa = MFA.MFA_source_separation(net, FFT_dict=FFT_dict, args=args)

    # 用自己分析 latent matrix，新版
    l, h = LA.get_binearlization_latent_matrix()
    plt.clf()
    plt.suptitle("Otsu thresholding")
    fig, (latent_repre, l_plot, h_plot) = plt.subplots(1, 3)
    latent_repre.set_title('latent representation', pad=24)
    latent_repre.imshow(l+h)
    l_plot.set_title('l', pad=24)
    l_plot.imshow(l)
    h_plot.set_title('h', pad=24)
    h_plot.imshow(h)
    plt.suptitle("Representation split")
    plt.tight_layout()
    # 取得格林威治偏移秒
    _glin = str(int(time.mktime(time.localtime())))
    os.makedirs(f"latent_representation_ana", exist_ok=True)
    plt.savefig(f'{args.logdir}/plot_figures/latent_representation_ana.png')  # hard-code
    plt.savefig(f'./latent_representation_ana/{_glin}.png')
    #plt.show()

    # 使用固定 thres
    # fix_thres = LA.get_otsu_threshold()
    # l, h = LA.get_binearlization_latent_matrix_by_fix_threshold(fix_thres,
    #                                                             delta=4,
    #                                                             plot=True,
    #                                                             plot_otsu=True)

    # 將資料注入到 mfa 物件去做繪製。
    mfa.low_thersh_encode_img = torch.tensor(l.T, device='cuda')
    mfa.high_thersh_encode_img = torch.tensor(h.T, device='cuda')

    for test_file in test_filelist:
        # load test data
        lps, phase, mean, std = val_dataloader(test_file, FFT_dict)
        mfa.source_separation(np.array(lps), np.array(phase),
                              np.array(mean), np.array(std),
                              filedir=outdir,
                              filename=test_filename)
