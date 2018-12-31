#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import cv2
import time
import numpy as np
import argparse
from PIL import Image

from faceboxes import FaceBox

from data.config import cfg
from torch.autograd import Variable
from utils.augmentations import FaceBoxesBasicTransform


parser = argparse.ArgumentParser(description='faceboxes demo')
parser.add_argument('--save_dir',
                    type=str, default='tmp/',
                    help='Directory for detect result')
parser.add_argument('--model',
                    type=str,
                    default='weights/faceboxes.pth', help='trained model')
parser.add_argument('--thresh',
                    default=0.5, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect(net, img_path, thresh):
     #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape

    x = FaceBoxesBasicTransform(img)

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    t1 = time.time()
    y = net(x)
    detections = y.data

    scale = torch.Tensor([width, height, width, height])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    t3 = time.time()
    for i in range(detections.size(1)):
        for j in range(detections.size(2)):
            if detections[0, i, j, 0] >= args.thresh:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] *
                      scale).cpu().numpy().astype(int)
                left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
                cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
                conf = "{:.2f}".format(score)
                text_size, baseline = cv2.getTextSize(
                    conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                p1 = (left_up[0], left_up[1] - text_size[1])
                cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                              (p1[0] + text_size[0], p1[1] + text_size[1]), [255, 0, 0], -1)
                cv2.putText(img, conf, (p1[0], p1[
                            1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))

    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


if __name__ == '__main__':
    net = FaceBox(cfg, 'test')
    net.load_state_dict(torch.load(args.model))
    net.eval()
    if use_cuda:
        net = net.cuda()
        net.benckmark = True

    img_path = './img'
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]

    for path in img_list:
        detect(net, path, args.thresh)
