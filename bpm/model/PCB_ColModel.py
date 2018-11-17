import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from .resnet_conv1 import resnet50


class PCBModel(nn.Module):
  def __init__(
      self,
      last_conv_stride=1,
      last_conv_dilation=1,
      num_stripes=6,                     
      local_conv_out_channels=256,
      num_classes=0,
      num_cols=1,
      rpp=True
  ):
    super(PCBModel, self).__init__()

    self.base = resnet50(
      pretrained=True,
      last_conv_stride=last_conv_stride,
      last_conv_dilation=last_conv_dilation)
    self.num_stripes = num_stripes
    self.num_cols = num_cols

    self.local_conv_list = nn.ModuleList()
    for _ in range(num_stripes*num_cols):
      self.local_conv_list.append(nn.Sequential(
        nn.Conv2d(2048, local_conv_out_channels, 1),
        nn.BatchNorm2d(local_conv_out_channels),
        nn.ReLU(inplace=True)
      ))
    if num_classes > 0:
      self.fc_list = nn.ModuleList()
      for _ in range(num_stripes*num_cols):
        fc = nn.Linear(local_conv_out_channels, num_classes)
        init.normal(fc.weight, std=0.001)
        init.constant(fc.bias, 0)
        self.fc_list.append(fc)

  def forward(self, x):
    """
    Returns:
      local_feat_list: each member with shape [N, c]
      logits_list: each member with shape [N, num_classes]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    assert feat.size(2) % self.num_stripes == 0
    stripe_h = int(feat.size(2) / self.num_stripes)
    col_w = int(feat.size(3) / self.num_cols)
    # print('col_w:{}'.format(col_w))
    local_feat_list = []
    logits_list = []
    g_list = []
    h_list = []
    m = 0
    for i in range(self.num_stripes):
      for j in range(self.num_cols):
        # print(feat.shape)
        local_feat_t = F.avg_pool2d(
        feat[:, :, i * stripe_h: (i + 1) * stripe_h, j*col_w:(j+1)*col_w],
        (stripe_h, col_w))
        # print(feat[:, :, i * stripe_h: (i + 1) * stripe_h, j*col_w:(j+1)*col_w].shape)
        # print(local_feat_t.shape)
        
        local_feat_g = self.local_conv_list[m](local_feat_t)
        # print(local_feat_g.shape)
        local_feat_g = local_feat_g.view(local_feat_g.size(0), -1)
        local_feat_list.append(local_feat_g)
        if hasattr(self,'fc_list'):
          logits_list.append(self.fc_list[m](local_feat_g))
        # print('m:{}'.format(m))
        m = m+1
    
    if hasattr(self, 'fc_list'):
      
      # get the 12 parts
      return local_feat_list,logits_list

    return local_feat_list

