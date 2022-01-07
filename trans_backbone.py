import mmcv
from argparse import ArgumentParser

from mmdet3d.models import build_model
import torch
from torch import nn
from mmdet3d.datasets.pipelines import Compose
import numpy as np
import onnx
import onnxruntime
from copy import deepcopy
from mmcv.parallel import collate, scatter

from mmdet.models.builder import (BACKBONES, HEADS, NECKS)
from mmdet.core import build_bbox_coder
from mmdet3d.core import anchor, bbox3d2result, build_prior_generator, xywhr2xyxyr, box3d_multiclass_nms, limit_period
from mmdet3d.core.bbox import get_box_type, LiDARInstance3DBoxes
from mmdet3d.apis import init_model

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pts_backbone = BACKBONES.build(cfg['pts_backbone'])
        self.pts_neck = NECKS.build(cfg['pts_neck'])
        self.cfg = cfg
        pts_bbox_head = cfg['pts_bbox_head']
        train_cfg = cfg['train_cfg']
        test_cfg = cfg['test_cfg']
        pts_train_cfg = train_cfg.pts if train_cfg else None
        pts_bbox_head.update(train_cfg=pts_train_cfg)
        pts_test_cfg = test_cfg.pts if test_cfg else None
        pts_bbox_head.update(test_cfg=pts_test_cfg)
        self.pts_bbox_head = HEADS.build(pts_bbox_head)
        self.anchor_generator = build_prior_generator(cfg['pts_bbox_head']['anchor_generator'])
        self.bbox_coder = build_bbox_coder(cfg['pts_bbox_head']['bbox_coder'])
        self.box_code_size = self.bbox_coder.code_size
        loss_cls = cfg['pts_bbox_head']['loss_cls']
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.dir_limit_offset = cfg['pts_bbox_head']['dir_limit_offset']
        self.dir_offset = cfg['pts_bbox_head']['dir_offset']

    def forward(self, input):
        x = input[:64*400*400]
        x = x.reshape(-1, 64, 400, 400)
        anchors = input[64*400*400:]
        anchors = anchors.reshape(-1, 9)

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        x = self.pts_bbox_head(x)
        bboxes, scores, labels = self.get_boxes(*x, anchors)
        return bboxes, scores, labels
        # return x, anchors
    
    def decode(self, anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.
        """
        cas, cts = [], []
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

    def get_boxes(self, cls_scores, bbox_preds, dir_cls_preds, anchors):
        cls_score = cls_scores[0].detach()
        bbox_pred = bbox_preds[0].detach()
        dir_cls_pred = dir_cls_preds[0].detach()
        cls_score = cls_scores[0].detach()
        bbox_pred = bbox_preds[0].detach()
        dir_cls_pred = dir_cls_preds[0].detach()
        bboxes, scores, labels = self.get_bboxes_single(cls_score, bbox_pred,
                                            dir_cls_pred, anchors)
        return bboxes, scores, labels

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_cls_preds, anchors):
        cls_score = cls_scores[0]
        bbox_pred = bbox_preds[0]
        dir_cls_pred = dir_cls_preds[0]
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.pts_bbox_head.num_classes)
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(1, 2,
                                        0).reshape(-1, self.box_code_size)

        nms_pre = 1000
        if self.use_sigmoid_cls:
            max_scores, _ = scores.max(dim=1)
        else:
            max_scores, _ = scores[:, :-1].max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        topk_inds = topk_inds.long()
        anchors = anchors[topk_inds, :]
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        dir_cls_score = dir_cls_score[topk_inds]
        bboxes = self.decode(anchors, bbox_pred)
        return bboxes, scores, dir_cls_score

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

def preprocess(model, pcd):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['points'] = data['points'][0].data
    return data

def build_backbone_model(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    model = Backbone(config.model)
    model.to('cuda').eval()

    checkpoint = torch.load(checkpoint, map_location='cuda')
    dicts = {}
    for key in checkpoint["state_dict"].keys():
        if "backbone" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "neck" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "bbox_head" in key:
            dicts[key] = checkpoint["state_dict"][key]
    model.load_state_dict(dicts)

    torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # parse_model(model)
    data = preprocess(model, args.pcd)
    img_metas = data['img_metas'][0]
    pts = data['points'][0]

    backbone_model = build_backbone_model(args.config, args.checkpoint, device=args.device)
    # parse_model(backbone_model)

    # original_model forward
    voxels, num_points, coors = model.voxelize(pts)
    voxel_features, raw_feats = model.pts_voxel_encoder(voxels, num_points, coors,
                                            None, img_metas)

    batch_size = coors[-1, 0] + 1
    feature_map = model.pts_middle_encoder(voxel_features, coors, batch_size)

    anchors = np.load('./np_anchors.npy')
    anchors = torch.from_numpy(anchors).cuda("cuda:1")
    # print("-----------------ready to export onnx --------------------------")
    rfeature_map = feature_map.reshape(-1)
    ranchors = anchors.reshape(-1)
    input = torch.cat((rfeature_map, ranchors))

    # export to onnx
    export_onnx_file = './pts_backbone.onnx'
    torch.onnx.export(backbone_model,
                    input,
                    export_onnx_file,
                    opset_version=12,
                    verbose=True,
                    do_constant_folding=True) # 输出名

if __name__ == '__main__':
    main()