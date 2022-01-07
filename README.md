# mmdet3d_onnx_tools
convert hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d from mmdet3d to onnx
## Installation
1. first install the official [MMdetection3D](https://github.com/open-mmlab/mmdetection3d)
2. download the official checkpoint file : hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth
3. run srcipts:
```bash
  python trans_vfe.py ./n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin ~/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py ~/mmdetection3d/checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth
  python trans_backbone.py ./n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin ~/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py ~/mmdetection3d/checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth
```
