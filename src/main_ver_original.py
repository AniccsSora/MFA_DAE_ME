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
from dataset.HLsep_dataloader import hl_dataloader, val_dataloader
import scipy.io.wavfile as wav
import os
from os.path import join as pjoin
import tester
import logging
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
parser.add_argument('--CosineAnnealingWarmRestarts', type=bool, default=False, help='optimizer scheduler for training')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 10)')
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
        "encoder": [32, 16, 8],
        "decoder": [8, 16, 32, 1],
        "encoder_filter": [[1, 3], [1, 3], [1, 3]],
        "decoder_filter": [[1, 3], [1, 3], [1, 3], [1, 1]],
        "encoder_act": "relu",
        "decoder_act": "relu",
        "dense": [],
        }
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
    'FFTSize': 2048,
    'Hop_length': 128,
    'Win_length': 2048,
    'normalize': True,
}
# declare model object
net = Model[args.model_type](model_dict=model_dict[args.model_type], args=args, logger=logger).cuda()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    net.cuda()

if __name__ == "__main__":

    # data loader
    test_filelist = ["./dataset/4_1.wav"]
    test_filename = test_filelist[0].split('/')[-1].split('.')[0]  # get pure-filename
    outdir = "{}/test_".format(args.logdir)
    train_loader = hl_dataloader(test_filelist,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=False,
                                 FFT_dict=FFT_dict,
                                 args=args)
    # train
    net = train.train(train_loader, net, args, logger)

    # Source Separation by MFA analysis.
    mfa = MFA.MFA_source_separation(net, FFT_dict=FFT_dict, args=args)
    for test_file in test_filelist:
        # load test data
        lps, phase, mean, std = val_dataloader(test_file, FFT_dict)
        mfa.source_separation(np.array(lps), np.array(phase),
                              np.array(mean), np.array(std),
                              filedir=outdir,
                              filename=test_filename)
