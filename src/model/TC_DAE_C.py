# -*- coding: utf-8 -*-
#Author:Wei-Chien Wang

import numpy as np
import os
import torch
import torch.nn as nn
import math


def ACT(act_f):
   if(act_f=="relu"):
       return torch.nn.ReLU()
   elif(act_f=="tanh"):
       return torch.nn.Tanh()
   elif(act_f=="relu6"):
       return nn.ReLU6()
   elif(act_f=="sigmoid"):
       return nn.Sigmoid()
   elif(act_f=="LeakyReLU"):
       return nn.LeakyReLU()
   elif(act_f=="ELU"):
       return nn.ELU()
   else:
       print("Doesn't support {0} activation function".format(act_f))


class Encoder(nn.Module):


    def __init__(self, model_dict=None, padding="same", args=None, logger=None):
        super(Encoder, self).__init__()
        self.model_dict = model_dict
        self.padding = padding
        self.feature_dim = self.model_dict['frequency_bins'][1] - self.model_dict['frequency_bins'][0]
        self.encoder_act = self.model_dict['encoder_act']
        self.encoder_layer = self.model_dict['encoder']
        self.encoder_filter = self.model_dict['encoder_filter']
        self.conv_layers = self._make_layers()


    def _make_layers(self):

        layers = []
        in_channels = self.feature_dim

        for i in range(0, len(self.encoder_layer)):
            out_channels = self.encoder_layer[i]
            encoder_layer = nn.Conv2d(in_channels, out_channels, kernel_size = (1, 1), stride = (1, 1), padding = 0, bias = True)
            bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
            in_channels = out_channels

            layers.append(encoder_layer)
            layers.append(bn)
            layers.append(ACT(self.encoder_act))
        return nn.Sequential(*layers)


    def forward(self, x):
        # (1, F, T, 1) Temporal Convolutional
        x = self.conv_layers(x)
        # transpose
        return x

class Decoder(nn.Module):


    def __init__(self, model_dict=None, padding="same", args=None, logger=None):
        super(Decoder, self).__init__()
        self.model_dict = model_dict
        self.padding = padding
        self.feature_dim = self.model_dict['frequency_bins'][1] - self.model_dict['frequency_bins'][0]
        self.encoder_act = self.model_dict['encoder_act']
        self.encoder_layer = self.model_dict['encoder']
        self.encoder_filter = self.model_dict['encoder_filter']
        self.decoder_layer = self.model_dict['decoder']
        self.decoder_filter = self.model_dict['decoder_filter']
        self.conv_layers = self._make_layers()


    def _make_layers(self):

        layers = []
        in_channels =  self.encoder_layer[-1]

        for i in range(0, len(self.decoder_layer)):
            out_channels = self.decoder_layer[i]
            decoder_layer = nn.ConvTranspose2d(in_channels, out_channels, (1, 1), stride=(1, 1), padding=0, bias= True)
            bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
            in_channels = out_channels

            layers.append(decoder_layer)
            layers.append(bn)
            layers.append(ACT(self.encoder_act))
        return nn.Sequential(*layers)


    def forward(self, x):
        # (1, F, T, 1) Temporal Convolutional
        x = self.conv_layers(x)
        # transpose
        return x



class TC_DAE_C(nn.Module):

    """
    # deep convolutional autoencoder
    """

    def __init__(self, model_dict=None, padding="same", args=None, logger=None):
        super(TC_DAE_C, self).__init__()
        if model_dict==None:
            self.model_dict = {
                "frequency_bins": [0, 257],
                "encoder": [512, 256, 128],
                "decoder": [256, 512, 257],
                "encoder_filter": [[1, 1],[1, 1], [1, 1]],
                "decoder_filter": [[1, 1],[1, 1], [1, 1]],
                "encoder_act": "ELU",
                }
        else:
            self.model_dict = model_dict
        self.encoder = Encoder(self.model_dict)
        self.decoder = Decoder(self.model_dict)


        
    def forward(self, x):

        # input (1, F, T, 1)
        y_t = self.encoder(x)
        print(y_t.shape)
        #latent shape
        latent = torch.reshape(y_t, (y_t.shape[1], y_t.shape[2])) # (1, T, F*C)
        print(latent.shape)
        latent = torch.reshape(latent, (1, y_t.shape[1], y_t.shape[2], 1)) # (1, T, F*C)
        print(latent.shape)
        output = self.decoder(latent)


        #output = torch.cat((x_c, x_n), 1)
        output = torch.permute(output, (0, 2, 1, 3))
        output = output.view(-1, output.shape[1], 257)

        return output


if __name__=="__main__":
    #####test phasea
    x = torch.tensor(np.ones((1, 257, 606, 1))).float()

    """
    model_dict = {
        "frequency_bins":[0, 257],
        "encoder":[1024, 512, 256, 32],
        "decoder":[32, 256, 512, 1024, 257],
        "encoder_filter":[[1,1],[1,1],[1,1],[1,1]],
        "decoder_filter":[[1,1],[1,1],[1,1],[1,1],[1,1]],
        "encoder_act":"relu",
        "decoder_act":"relu",
        "dense":[16],
        }
    encoder = Encoder(model_dict)
    decoder = Decoder(model_dict)
    print(x.shape)
    output = encoder(x)
    print(output.shape)
    x_ = decoder(output)
    print(x_.shape)
    """
    
    model = TC_DAE_C()
    y = model(x)
