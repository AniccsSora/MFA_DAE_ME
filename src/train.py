#/--coding:utf-8/
import os
import time
from datetime import datetime
import numpy as np
import torch
from os.path import join as pjoin
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import misc, make_path, signalprocess
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(train_loader, net=None, args=None, logger=None):
    best_acc, old_file = 0, None
    per_save_epoch = 30
    # optimizer
    if args.optim=="RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    elif args.optim=="Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    # loss function
    mse = nn.MSELoss()
    if args.CosineAnnealingWarmRestarts:
        print("cos")
        train_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, verbose=True)

    # scheduler_RLROP = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
    #                                                  patience=5,
    #                                                 verbose=True)
    figure_recoder_loss = []
    figure_recoder_lr = []
    old_file = 0
    __loss = None
    # start training
    for epoch in tqdm(range(args.epochs), desc='Training epoch', smoothing=0.1, ncols=100):
        avg_batch_loss = 0
        for batch_idx, data in enumerate(train_loader):
            if args.cuda:
                data = data.cuda().float()
            data = Variable(data)
            optimizer.zero_grad()
            output = net(data)  
            loss = mse(output, data)
            avg_batch_loss +=loss
            loss.backward()
            optimizer.step()
            if args.CosineAnnealingWarmRestarts:
                train_scheduler.step(epoch+batch_idx/len(train_loader))

        new_file = os.path.join(args.logdir, 'latest.pt')
        misc.model_save(net, new_file, old_file=old_file, verbose=False)
        old_file = new_file

        __loss = avg_batch_loss / batch_idx
        #scheduler_RLROP.step(__loss)
        logger("epoch{0}:{1}".format(epoch, __loss))
        figure_recoder_loss.append(__loss.detach().cpu().item())
        figure_recoder_lr.append(train_scheduler.get_last_lr()[0])

    # 紀錄 最後的 loss
    with open(pjoin(args.logdir, 'final_loss_{:.4f}'.format(__loss)), mode='w'):
        print("final loss:", __loss.item())
    plt.plot(figure_recoder_loss, label='loss')
    plt.plot(figure_recoder_lr, alpha=0.5, label='lr')
    plt.yscale('log')
    last_points = (figure_recoder_loss[-1], figure_recoder_lr[-1])
    ax = plt.gca()
    ax.annotate("{:.4f}".format(last_points[0]), (args.epochs, last_points[0]), textcoords='offset pixels', xytext=(30, -8))
    ax.annotate("{:.4f}".format(last_points[1]), (args.epochs, last_points[1]), textcoords='offset pixels', xytext=(30, -8))
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(args.logdir, 'loss.png'))

    return net
