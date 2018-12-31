#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable
from layers import PriorBox
from layers import Detect
from data.config import cfg
import time


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


class Inception(nn.Module):

    def __init__(self):
        super(Inception, self).__init__()
        self.conv1 = conv_bn_relu(128, 32, kernel_size=1)
        self.conv2 = conv_bn_relu(128, 32, kernel_size=1)
        self.conv3 = conv_bn_relu(128, 24, kernel_size=1)
        self.conv4 = conv_bn_relu(24, 32, kernel_size=3, padding=1)
        self.conv5 = conv_bn_relu(128, 24, kernel_size=1)
        self.conv6 = conv_bn_relu(24, 32, kernel_size=3, padding=1)
        self.conv7 = conv_bn_relu(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x2 = self.conv2(x2)

        x3 = self.conv3(x)
        x3 = self.conv4(x3)

        x4 = self.conv5(x)
        x4 = self.conv6(x4)
        x4 = self.conv7(x4)

        output = torch.cat([x1, x2, x3, x4], 1)
        return output


class MultiBoxLayer(nn.Module):
    num_classes = 2
    num_anchors = [21, 1, 1]
    in_planes = [128, 256, 256]

    def __init__(self):
        super(MultiBoxLayer, self).__init__()

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[
                                   i] * 4, kernel_size=3, padding=1))
            classs_num = 2
            # if i==0:
            #    classs_num = 4
            self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[
                                    i] * classs_num, kernel_size=3, padding=1))

    def forward(self, xs):
        '''
        xs:list of 之前的featuremap list
        retrun: loc_preds: [N,21824,4]  21284 = 32*32*(4*4+2*2+1)+16*16+8*8
                        conf_preds:[N,21824,2]

        '''
        y_locs = []
        y_confs = []
        for i, x in enumerate(xs):
            y_loc = self.loc_layers[i](x)  # N,anhors*4,H,W
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            '''
            if i==0:
                y_conf = y_conf.view(N,-1,4)
                bg_max_out,_ = torch.max(y_conf[:,:,0:3],dim=-1,keepdim=True)
                y_conf = torch.cat((bg_max_out,y_conf[:,:,3:]),dim=-1)
            else:
            	y_conf = y_conf.view(N, -1, 2)'''
            y_conf = y_conf.view(N, -1, 2)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)

        return loc_preds, conf_preds


class FaceBox(nn.Module):

    def __init__(self, cfg, phase='train'):
        super(FaceBox, self).__init__()
        self.phase = phase
        # model
        self.conv1 = nn.Conv2d(3, 24, kernel_size=7,
                               stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=5,
                               stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()

        self.conv3_1 = conv_bn_relu(128, 128, kernel_size=1)
        self.conv3_2 = conv_bn_relu(
            128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = conv_bn_relu(256, 128, kernel_size=1)
        self.conv4_2 = conv_bn_relu(
            128, 256, kernel_size=3, stride=2, padding=1)

        self.multilbox = MultiBoxLayer()

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.test_det = Detect(cfg)


    def forward(self, x):
        img_size = x.size()[2:]
        source = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(torch.cat((F.relu(x), F.relu(-x)), 1))

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(torch.cat((F.relu(x), F.relu(-x)), 1))

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        source.append(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        source.append(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        source.append(x)

        feature_maps = []
        for feat in source:
            feature_maps.append([feat.size(2), feat.size(3)])

        self.priors = Variable(PriorBox(img_size, feature_maps, cfg).forward())

        loc_preds, conf_preds = self.multilbox(source)

        if self.phase == 'test':
            output = self.test_det(loc_preds,
                                   self.softmax(conf_preds),
                                   self.priors)
        else:
            output = (
                loc_preds,
                conf_preds,
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def weights_init_body(self, m):
    	def gaussian(param):
    		init.normal(param,std=0.01)

        if isinstance(m, nn.Conv2d):
            gaussian(m.weight.data)
            if 'bias' in m.state_dict().keys():
            	m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()

    def weights_init_head(self, m):
        def xavier(param):
            init.xavier_uniform(param)

        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.fill_(0.2)


if __name__ == '__main__':
    net = FaceBox(cfg)
    print(net)
    inputs = Variable(torch.randn(1, 3, 1024, 1024))
    out = net(inputs)
    print(out[0].size())
