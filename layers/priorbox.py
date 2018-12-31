#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import torch
from itertools import product as product
from data.config import cfg


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, image_size,feature_maps,cfg):
        super(PriorBox, self).__init__()
        self.imh,self.imw = image_size[0],image_size[1]

        self.feature_maps = feature_maps
        # number of priors for feature map location (either 4 or 6)
        self.variance = cfg.VARIANCE or [0.1]

        self.steps = cfg.STEPS

        self.min_sizes = cfg.ANCHOR_SIZES
        self.aspect_ratios = cfg.ASPECT_RATIOS
        self.clip = cfg.CLIP
        self.density = cfg.DENSITY

        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k in range(len(self.feature_maps)):
            feath = self.feature_maps[k][0]
            featw = self.feature_maps[k][1]
            for h,w in product(range(feath),range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]
                 # unit center x,y
                cx = (w + 0.5) / f_kw
                cy = (h + 0.5) / f_kh

                # aspect_ratio: 1
                # rel size: min_size
                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh
                for j, ar in enumerate(self.aspect_ratios[k]):
                    if k == 0:
                        for dy, dx in product(self.density[j], repeat=2):
                            mean += [cx + dx / 8. * s_kw * ar, cy +
                                     dy / 8. * s_kh * ar, s_kw * ar, s_kh * ar]
                    else:
                        mean += [cx, cy, s_kw * ar, s_kh * ar]


        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    image_size = [1024,1024]
    feature_maps = [[32,32],[16,16],[8,8]]
    priors = PriorBox(image_size,feature_maps,cfg)
    output = priors.forward()
    print(output.size())
