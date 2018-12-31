#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np

import os
import time
import argparse

from data.config import cfg
from torch.autograd import Variable
from faceboxes import FaceBox
from layers import MultiBoxLoss
from data.wider_face import WIDERDetection, detection_collate


parser = argparse.ArgumentParser(
    description='faceboxes Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


train_dataset = WIDERDetection(cfg.TRAIN_FILE,
                               mode='train')

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)

val_dataset = WIDERDetection(cfg.VAL_FILE,
                             mode='val')
val_batch_size = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)
min_loss = np.inf


def train():
    start_epoch = 0
    per_epoch_size = len(train_dataset) // args.batch_size
    iteration = 0
    step_index = 0

    net = FaceBox(cfg)
    if use_cuda:
        net = net.cuda()
        cudnn.benckmark = True
    if not args.resume:
        print('Initializing weights...')
        net.apply(net.weights_init_body)
        net.multilbox.apply(net.weights_init_head)
    else:
        print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr,
    #                       weight_decay=args.weight_decay)

    criterion = MultiBoxLoss(cfg, use_cuda)
    print('Loading wider dataset...')
    print(args)

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        for batch_idx, (images, targets) in enumerate(train_loader):
            if use_cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()

            if iteration % 10 == 0:
                print('Timer:{:.4f}'.format(t1 - t0))
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + " || Loss:%.4f" % (loss.data[0]))
                print('->> loc loss:{:.4f} || conf loss:{:.4f}'.format
                      (loss_l.data[0], loss_c.data[0]))
                print('->> lr:{}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), os.path.join(args.save_folder, 'faceboxes_wider_' +
                                                          repr(iteration) + '.pth'))
            iteration += 1

        val(epoch, net, criterion)


def val(epoch, net, criterion):
    net.eval()
    step = 0
    t1 = time.time()
    losess = 0
    for batch_idx, (images, targets) in enumerate(val_loader):
        if use_cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        out = net(images)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        losess += loss.data[0]
        step += 1

    tloss = losess / step
    t2 = time.time()
    print('Timer:%.4f' % (t2 - t1))
    print('test Loss:{:.4f}'.format(tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        torch.save(net.state_dict(), os.path.join(
            args.save_folder, 'faceboxes.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': net.state_dict(),
    }
    torch.save(states, os.path.join(
        args.save_folder, 'faceboxes_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
