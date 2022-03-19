import argparse
import torch
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
parser.add_argument('--epochs', type=int, default=1 , help='number of epochs to train (default: 10)')
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
#
parser.add_argument('--time_convolution', type=bool, default=False)
#
parser.add_argument('--depthwiseConv', type=bool, default=False)
parser.add_argument('--depthwiseConv_K', type=int, default=2, help='activate when depthwiseConv is True')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

def get_args():
    return args


