# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import multiscale_supervision, geo_scal_loss, sem_scal_loss
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet.models.utils import build_transformer
from mmcv.cnn.utils.weight_init import constant_init
import os
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

@HEADS.register_module()
class OccHead(nn.Module): 
    def __init__(self,
                 *args,
                 transformer_template=None,
                 num_classes=17,
                 volume_h=200,
                 volume_w=200,
                 volume_z=16,
                 upsample_strides=[1, 2, 1, 2],
                 out_indices=[0, 2, 4, 6],
                 conv_input=None,
                 conv_output=None,
                 embed_dims=None,
                 img_channels=None,
                 use_semantic=True,
                 **kwargs):
        super(OccHead, self).__init__()
        self.conv_input = conv_input
        self.conv_output = conv_output
        
        
        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z

        self.img_channels = img_channels

        self.use_semantic = use_semantic
        self.embed_dims = embed_dims

        self.fpn_level = len(self.embed_dims) # 这个是不会变的参数
        self.upsample_strides = upsample_strides # 这个也是不会变的参数
        self.out_indices = out_indices # 这个也是不会变的参数
        self.transformer_template = transformer_template
        
        # 算了，不会变的也复制一份吧，保险起见 # mby # 这个还没弄
        
        self._init_layers() # 这里原来只初始化一次，现在要两次

    def _init_layers(self): # 拷贝一份这个构造
        self.transformer = nn.ModuleList()
        
        for i in range(self.fpn_level):
            transformer = copy.deepcopy(self.transformer_template)

            transformer.embed_dims = transformer.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points[i]

            transformer.encoder.transformerlayers.feedforward_channels = \
                self.transformer_template.encoder.transformerlayers.feedforward_channels[i]
            
            transformer.encoder.transformerlayers.embed_dims = \
                self.transformer_template.encoder.transformerlayers.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].embed_dims[i]
            
            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims[i]
            
            transformer.encoder.num_layers = self.transformer_template.encoder.num_layers[i]

            transformer_i = build_transformer(transformer)
            self.transformer.append(transformer_i)

        # 走到这里，transformer已经被初始化好了，深拷贝一份就行了，一定要深拷贝
        self.transformer_1 = copy.deepcopy(self.transformer) # mby

        self.deblocks = nn.ModuleList()
        upsample_strides = self.upsample_strides

        out_channels = self.conv_output
        in_channels = self.conv_input

        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg=dict(type='deconv3d', bias=False)
        conv_cfg=dict(type='Conv3d', bias=False)

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1)


            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))

            self.deblocks.append(deblock)

        # 走到这里self.deblocks已经初始化完毕了，深拷贝一份
        self.deblocks_1 = copy.deepcopy(self.deblocks)

        
        self.occ = nn.ModuleList()
        for i in self.out_indices:
            if self.use_semantic:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)
            else:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)

        # 走到这里self.occ已经初始化完毕了，深拷贝一份
        self.occ_1 = copy.deepcopy(self.occ)
        
        self.volume_embedding = nn.ModuleList()
        for i in range(self.fpn_level):
            self.volume_embedding.append(nn.Embedding(
                    self.volume_h[i] * self.volume_w[i] * self.volume_z[i], self.embed_dims[i]))

        # 走到这里self.volume_embedding已经初始化完毕了，深拷贝一份
        self.volume_embedding_1 = copy.deepcopy(self.volume_embedding)

        self.transfer_conv = nn.ModuleList()
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        conv_cfg=dict(type='Conv2d', bias=True)
        for i in range(self.fpn_level):
            transfer_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=self.img_channels[i],
                    out_channels=self.embed_dims[i],
                    kernel_size=1,
                    stride=1)
            transfer_block = nn.Sequential(transfer_layer,
                    nn.ReLU(inplace=True))

            self.transfer_conv.append(transfer_block)
        # 走到这里self.transfer_conv已经初始化完毕了，深拷贝一份
        self.transfer_conv_1 = copy.deepcopy(self.transfer_conv)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for i in range(self.fpn_level):
            self.transformer[i].init_weights()
        
        # modify by yufc
        for i in range(self.fpn_level):
            self.transformer_1[i].init_weights() # ?
        
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        # 这里弄一下
        # 没有_的是other, 有_1的是road的 # modify by yufc
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        volume_embed = [] # other # mby
        volume_embed_1 = [] # road # mby
        for i in range(self.fpn_level):
            volume_queries = self.volume_embedding[i].weight.to(dtype) # other # mby
            volume_queries_1 = self.volume_embedding_1[i].weight.to(dtype) # road # mby
            
            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]
            
            _, _, C, H, W = mlvl_feats[i].shape
            view_features = self.transfer_conv[i](mlvl_feats[i].reshape(bs*num_cam, C, H, W)).reshape(bs, num_cam, -1, H, W) # other # mby
            view_features_1 = self.transfer_conv_1[i](mlvl_feats[i].reshape(bs*num_cam, C, H, W)).reshape(bs, num_cam, -1, H, W) # road # mby
            
            volume_embed_i = self.transformer[i]( # other # mby
                [view_features],
                volume_queries,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                img_metas=img_metas
            )
            volume_embed_i_1 = self.transformer_1[i]( # road # mby
                [view_features],
                volume_queries_1,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                img_metas=img_metas
            )
            
            volume_embed.append(volume_embed_i)  # other # mby
            volume_embed_1.append(volume_embed_i_1)  # road # mby
            
            
        volume_embed_reshape = [] # other # mby
        volume_embed_reshape_1 = [] # road # mby
        
        for i in range(self.fpn_level):
            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]
            volume_embed_reshape_i = volume_embed[i].reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1) # other # mby
            volume_embed_reshape_i_1 = volume_embed_1[i].reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1) # road # mby
            
            volume_embed_reshape.append(volume_embed_reshape_i) # other # mby
            volume_embed_reshape_1.append(volume_embed_reshape_i_1) # other # mby
            
        outputs = [] # other # mby
        outputs_1 = [] # road # mby
        result = volume_embed_reshape.pop() # other # mby
        result_1 = volume_embed_reshape_1.pop() # road # mby
        
        # other # mby
        for i in range(len(self.deblocks)):
            result = self.deblocks[i](result)
            if i in self.out_indices:
                outputs.append(result)
            elif i < len(self.deblocks) - 2:  # we do not add skip connection at level 0
                volume_embed_temp = volume_embed_reshape.pop()
                result = result + volume_embed_temp
        
        # road # mby
        for i in range(len(self.deblocks_1)):
            result_1 = self.deblocks_1[i](result_1)
            if i in self.out_indices:
                outputs_1.append(result_1)
            elif i < len(self.deblocks_1) - 2:  # we do not add skip connection at level 0
                volume_embed_temp_1 = volume_embed_reshape_1.pop()
                result_1 = result_1 + volume_embed_temp_1
                
        
        occ_preds = [] # other # mby
        occ_preds_1 = [] # road # mby
        for i in range(len(outputs)):
            occ_pred = self.occ[i](outputs[i])
            occ_preds.append(occ_pred)

        for i in range(len(outputs_1)):
            occ_pred_1 = self.occ_1[i](outputs_1[i])
            occ_preds_1.append(occ_pred_1)

       # other # mby
        outs_other = {
            'volume_embed': volume_embed,
            'occ_preds': occ_preds,
        }
        
        # road # mby
        outs_road = {
            'volume_embed': volume_embed_1,
            'occ_preds': occ_preds_1,
        }

        return outs_other, outs_road


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_occ_other,
             preds_dicts_other,
             gt_occ_road,
             preds_dicts_road,  # mby
             img_metas):
        # other loss mby
        if not self.use_semantic:
            loss_dict_other = {}
            for i in range(len(preds_dicts_other['occ_preds'])):
                pred = preds_dicts_other['occ_preds'][i][:, 0]
                ratio = 2**(len(preds_dicts_other['occ_preds']) - 1 - i)
                gt = multiscale_supervision(gt_occ_other.clone(), ratio, preds_dicts_other['occ_preds'][i].shape)
                #gt = torch.mode(gt, dim=-1)[0].float()
                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + geo_scal_loss(pred, gt.long(), semantic=False))      
                loss_occ_i =  loss_occ_i * ((0.5)**(len(preds_dicts_other['occ_preds']) - 1 -i)) #* focal_weight
                loss_dict_other['loss_occ_{}'.format(i)] = loss_occ_i
        else:
            pred = preds_dicts_other['occ_preds']  
            criterion = nn.CrossEntropyLoss(
                ignore_index=255, reduction="mean"
            )
            loss_dict_other = {}
            for i in range(len(preds_dicts_other['occ_preds'])):
                pred = preds_dicts_other['occ_preds'][i]
                ratio = 2**(len(preds_dicts_other['occ_preds']) - 1 - i)    
                gt = multiscale_supervision(gt_occ_other.clone(), ratio, preds_dicts_other['occ_preds'][i].shape)
                loss_occ_i = (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
                loss_occ_i = loss_occ_i * ((0.5)**(len(preds_dicts_other['occ_preds']) - 1 -i))
                loss_dict_other['loss_occ_{}'.format(i)] = loss_occ_i

        # road loss mby
        if not self.use_semantic:
            loss_dict_road = {}
            for i in range(len(preds_dicts_road['occ_preds'])):
                pred = preds_dicts_road['occ_preds'][i][:, 0]
                ratio = 2**(len(preds_dicts_road['occ_preds']) - 1 - i)
                gt = multiscale_supervision(gt_occ_road.clone(), ratio, preds_dicts_road['occ_preds'][i].shape)
                #gt = torch.mode(gt, dim=-1)[0].float()
                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + geo_scal_loss(pred, gt.long(), semantic=False))      
                loss_occ_i =  loss_occ_i * ((0.5)**(len(preds_dicts_road['occ_preds']) - 1 -i)) #* focal_weight
                loss_dict_road['loss_occ_{}'.format(i)] = loss_occ_i
        else:
            pred = preds_dicts_road['occ_preds']  
            criterion = nn.CrossEntropyLoss(
                ignore_index=255, reduction="mean"
            )
            loss_dict_road = {}
            for i in range(len(preds_dicts_road['occ_preds'])):
                pred = preds_dicts_road['occ_preds'][i]
                ratio = 2**(len(preds_dicts_road['occ_preds']) - 1 - i)    
                gt = multiscale_supervision(gt_occ_road.clone(), ratio, preds_dicts_road['occ_preds'][i].shape)
                loss_occ_i = (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
                loss_occ_i = loss_occ_i * ((0.5)**(len(preds_dicts_road['occ_preds']) - 1 -i))
                loss_dict_road['loss_occ_{}'.format(i)] = loss_occ_i
                    
        return loss_dict_other, loss_dict_road

        
