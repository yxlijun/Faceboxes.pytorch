#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
sys.path.append('/home/lj/face_detection/faceboxes')
sys.path.append(os.path.abspath(__file__))
import torch
import cv2
import time
import numpy as np
import argparse

from faceboxes import FaceBox
from PIL import Image

from data.config import cfg
from torch.autograd import Variable
from utils.augmentations import FaceBoxesBasicTransform


parser = argparse.ArgumentParser(description='faceboxes demo')

parser.add_argument('--model',
                    type=str,
                    default='weights/faceboxes.pth', help='trained model')
parser.add_argument('--thresh',
                    default=0.05, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


AFW_IMG_DIR = os.path.join(cfg.AFW_DIR, 'images')
AFW_RESULT_DIR = os.path.join(cfg.AFW_DIR, 'faceboxes')
AFW_RESULT_IMG_DIR = os.path.join(AFW_RESULT_DIR, 'images')

if not os.path.exists(AFW_RESULT_IMG_DIR):
    os.makedirs(AFW_RESULT_IMG_DIR)


def detect_face(net, img_path, thresh):
    h, w, _ = img.shape
    im_scale = 1.0
    if im_scale != 1.0:
        image = cv2.resize(img, None, None, fx=im_scale,
                           fy=im_scale, interpolation=cv2.INTER_LINEAR).copy()
    else:
        image = img
    x = FaceBoxesBasicTransform(image)
    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    t1 = time.time()
    y = net(x)
    detections = y.data

    scale = torch.Tensor([w, h, w, h])

    bboxes = []
    for i in range(detections.size(1)):
        for j in range(detections.size(2)):
            if detections[0, i, j, 0] >= thresh:
                box = []
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] *
                      scale).cpu().numpy().astype(np.int)
                box += [pt[0], pt[1], pt[2], pt[3], score]
                box[1] += 0.2 * (box[3] - box[1] + 1)
                bboxes += [box]
            else:
                break
    return bboxes

if __name__ == '__main__':
    net = FaceBox(cfg, 'test')
    net.load_state_dict(torch.load(args.model))
    net.eval()
    if use_cuda:
        net = net.cuda()
        net.benckmark = True

    counter = 0
    txt_out = os.path.join(AFW_RESULT_DIR, 'faceboxes_dets.txt')
    txt_in = os.path.join('./tools/afw_img_list.txt')

    fout = open(txt_out, 'w')
    fin = open(txt_in, 'r')

    for line in fin.readlines():
        line = line.strip()
        img_file = os.path.join(AFW_IMG_DIR, line + '.jpg')
        out_file = os.path.join(AFW_RESULT_IMG_DIR, line + '.jpg')
        counter += 1
        t1 = time.time()
        #img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = Image.open(img_file)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = np.array(img)
        bboxes = detect_face(net, img, args.thresh)
        t2 = time.time()
        print('Detect %04d th image costs %.4f' % (counter, t2 - t1))
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            fout.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                line, score, x1, y1, x2, y2))
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(out_file, img)

    fout.close()
    fin.close()
