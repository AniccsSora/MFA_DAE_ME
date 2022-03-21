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


def non_zero_loss(latent, device, alpha, tolerate):
    #with torch.no_grad():
    latent = latent.view(-1, )
    # latent 批次 總元素量
    total_size = latent.size(dim=0)
    # latetn 批次 為 0 的元素數量。
    is_zero = len((latent == 0).nonzero())
    is_zero = is_zero-(tolerate*total_size)
    is_zero = is_zero if is_zero > 0 else 0.0
    loss = alpha * (is_zero/total_size)
    #print(f"loss: {loss}")
    return loss

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
        # T_0: 何時執行第一次重啟。
        # T_mult: 後續的重啟時間的乘法因子
        alpha_ = args.epochs/50
        the_first_restart = int(args.epochs*(0.08/alpha_))
        T_mult = 2
        print(f"CosineAnnealingWarmRestarts: T_0:{the_first_restart}, T_mult:{T_mult}")
        train_scheduler = optim.lr_scheduler.\
            CosineAnnealingWarmRestarts(optimizer,
                                        T_0=the_first_restart,
                                        T_mult=T_mult, verbose=True)

    # scheduler_RLROP = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
    #                                                  patience=5,
    #                                                 verbose=True)
    figure_recoder_loss = []
    figure_recoder_loss_beta = []
    beta_loss = False  # 使用自己設計的額外 loss function
    figure_recoder_lr = []
    old_file = 0
    __loss = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # start training
    for epoch in tqdm(range(args.epochs), desc='Training epoch', smoothing=0.1, ncols=100):
        avg_batch_loss = 0
        for batch_idx, data in enumerate(train_loader):
            if args.cuda:
                data = data.cuda().float()
            data = Variable(data)
            optimizer.zero_grad()
            output = net(data)
            if beta_loss:
                latent = net.encoder(data)
                loss_beta = non_zero_loss(latent, device,
                                          alpha=1.0,  # 超參數
                                          tolerate=0.0)  # 允許多少比例的 latent represent 為 0
                figure_recoder_loss_beta.append(loss_beta)
            if beta_loss:
                loss = mse(output, data) + loss_beta
            else:
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
    pt_1, pt_2 = last_points[0], last_points[1]  # 排序，較高(數值較大)的先畫
    if last_points[1] > last_points[0]:
        pt_1, pt_2 = pt_2, pt_1
    ax.annotate("{:.4f}".format(pt_1),
                xy=(args.epochs, pt_1),
                textcoords='offset pixels', xytext=(120, 200),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.3",
                                ),
                )
    ax.annotate("{:.1e}".format(pt_2),
                xy=(args.epochs, pt_2),
                textcoords='offset pixels', xytext=(120, -200),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=0.3",
                                ),
                )
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(pjoin(args.logdir, 'loss.png'))
    #
    # 繪製 額外的 loss
    if beta_loss:
        plt.clf()
        plt.yscale('log')
        plt.plot(figure_recoder_loss_beta)
        plt.tight_layout()
        plt.savefig(pjoin(args.logdir, 'loss_beta.png'))

    return net
