from ctypes import resize
import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS

@NECKS.register_module()
class YOLOsimpNeck(BaseModule):


    def __init__(self,
                 in_channels,
                 out_channels,
                 num_scales=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(YOLOsimpNeck, self).__init__(init_cfg)
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        for i in range(self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            self.add_module(f'conv{i}', ConvModule(in_c, out_c, 1,**cfg))
            self.add_module(f'ponv{i}', ConvModule(out_c, out_c, 3, padding=1, **cfg))
            self.add_module(f'ponv2{i}', ConvModule(out_c, out_c, 3, padding=1, **cfg))


    def forward(self, feats):
        assert len(feats) == self.num_scales
        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        
        for i, x in enumerate(feats):
            #print(x.shape)
            x1 = resize(x,(94, 311))
            #print(x1.shape)
            conv = getattr(self, f'conv{i}')
            tmp = conv(x1)
            conv1 = getattr(self, f'ponv{i}')
            out = conv1(tmp)
            conv2 = getattr(self, f'ponv2{i}')
            out = conv2(out)
            outs.append(out)
           
        tmp = torch.cat(outs,dim=1)
        return tuple([tmp])
