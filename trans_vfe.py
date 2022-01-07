import mmcv
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16,force_fp32
from mmcv.runner import load_checkpoint
from argparse import ArgumentParser

from mmdet3d.models import build_model
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from mmcv.parallel import collate, scatter
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type
import numpy as np

class VFELayer(nn.Module):
    """Voxel Feature Encoder layer.

    The voxel encoder is composed of a series of these layers.
    This module do not support average pooling and only support to use
    max pooling to gather features inside a VFE.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        max_out (bool): Whether aggregate the features of points inside
            each voxel and only return voxel features.
        cat_max (bool): Whether concatenate the aggregated features
            and pointwise features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 max_out=True,
                 cat_max=True):
        super(VFELayer, self).__init__()
        self.fp16_enabled = False
        self.cat_max = cat_max
        self.max_out = max_out
        # self.units = int(out_channels / 2)

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.

        Returns:
            torch.Tensor: Voxel features. There are three mode under which the
                features have different meaning.
                - `max_out=False`: Return point-wise features in
                    shape (N, M, C).
                - `max_out=True` and `cat_max=False`: Return aggregated
                    voxel features in shape (N, C)
                - `max_out=True` and `cat_max=True`: Return concatenated
                    point-wise features in shape (N, M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]
        if self.max_out:
            aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        else:
            # this is for fusion layer
            return pointwise

        if not self.cat_max:
            return aggregated.squeeze(1)
        else:
            # [K, 1, units]
            repeated = aggregated.repeat(1, voxel_count, 1)
            concatenated = torch.cat([pointwise, repeated], dim=2)
            # [K, T, 2 * units]
            return concatenated

class VFE(nn.Module):
    def __init__(self,
                 cfg,
                 return_point_feats=False,
                 fusion_layer=None,
                 ):
        super().__init__()
        self.feat_channels = cfg.model['pts_voxel_encoder']['feat_channels']
        assert len(self.feat_channels) > 0
        self.in_channels = cfg.model['pts_voxel_encoder']['in_channels']
        self._with_distance = cfg.model['pts_voxel_encoder']['with_distance']
        self._with_cluster_center = cfg.model['pts_voxel_encoder']['with_cluster_center']
        self._with_voxel_center = cfg.model['pts_voxel_encoder']['with_voxel_center']
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False
        if self._with_cluster_center:
            self.in_channels += 3
        if self._with_voxel_center:
            self.in_channels += 3
        if self._with_distance:
            self.in_channels += 1

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = cfg['voxel_size'][0]
        self.vy = cfg['voxel_size'][1]
        self.vz = cfg['voxel_size'][2]
        self.point_cloud_range = cfg.model['pts_voxel_encoder']['point_cloud_range']
        self.x_offset = self.vx / 2 + self.point_cloud_range[0]
        self.y_offset = self.vy / 2 + self.point_cloud_range[1]
        self.z_offset = self.vz / 2 + self.point_cloud_range[2]
        self.norm_cfg = cfg.model['pts_voxel_encoder']['norm_cfg']

        feat_channels = [self.in_channels] + list(self.feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=self.norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)

    @force_fp32(out_fp16=True)
    def forward(self,voxel_feats):
        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)
        return voxel_feats

def parse_model(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])

def build_vfe_model(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    # original model
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint_load = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint_load['meta']:
            model.CLASSES = checkpoint_load['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names
        if 'PALETTE' in checkpoint_load['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint_load['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    parse_model(model)

    # VFE model
    pts_voxel_encoder = VFE(config)
    pts_voxel_encoder.to(device).eval()
    checkpoint_pts_load = torch.load(checkpoint, map_location=device)
    dicts = {}
    for key in checkpoint_pts_load["state_dict"].keys():
        if "vfe" in key:
            dicts[key[18:]] = checkpoint_pts_load["state_dict"][key]
    pts_voxel_encoder.load_state_dict(dicts)
    print('-----------------------')
    parse_model(pts_voxel_encoder)
    return model, pts_voxel_encoder

def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    model, pts_voxel_encoder = build_vfe_model(args.config, args.checkpoint, device=args.device)

    # export to onnx
    if isinstance(args.config, str):
        config = mmcv.Config.fromfile(args.config)
    dummy_input = torch.ones(config.model['pts_voxel_layer']['max_voxels'][1], config.model['pts_voxel_layer']['max_num_points'] , pts_voxel_encoder.in_channels).cuda()
    export_onnx_file = './pts_voxel_encoder.onnx'
    torch.onnx.export(pts_voxel_encoder,
                    dummy_input,
                    export_onnx_file,
                    opset_version=12,
                    verbose=True,
                    do_constant_folding=True) # 输出名

if __name__ == '__main__':
    main()