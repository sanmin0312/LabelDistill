# Copyright (c) Megvii Inc. All rights reserved.
"""

"""
from labeldistill.exps.base_cli import run_cli
from labeldistill.exps.nuscenes.base_exp import \
    LabelDistillModel as BaseLabelDistillModel
from labeldistill.models.labelencoder import LabelEncoder
from torch.optim.lr_scheduler import MultiStepLR

import torch

class LabelDistillModel(BaseLabelDistillModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = []

        #############################################################################################
        """
        Models:
          - Name: centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d
            In Collection: CenterPoint
            Config: configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
            metadata:
              Training Memory (GB): 5.2
            Results:
              - Task: 3D Object Detection
                Dataset: nuScenes
                Metrics:
                  mAP: 56.11
                  NDS: 64.61
            Weights: https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth
        """

        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        voxel_size = [0.1, 0.1, 0.2]

        bbox_coder = dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            code_size=9)

        train_cfg = dict(
            pts=dict(
                grid_size=[1024, 1024, 40],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))
        test_cfg = dict(
            pts=dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                nms_type='circle',
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2))

        self.lidar_conf = dict(type='CenterPoint',
                               pts_voxel_layer=dict(
                                   point_cloud_range=point_cloud_range, max_num_points=10, voxel_size=voxel_size,
                                   max_voxels=(90000, 120000)),
                               pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
                               pts_middle_encoder=dict(
                                   type='SparseEncoder',
                                   in_channels=5,
                                   sparse_shape=[41, 1024, 1024],
                                   output_channels=128,
                                   order=('conv', 'norm', 'act'),
                                   encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                                   encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
                                   block_type='basicblock'),
                               pts_backbone=dict(
                                   type='SECOND',
                                   in_channels=256,
                                   out_channels=[128, 256],
                                   layer_nums=[5, 5],
                                   layer_strides=[1, 2],
                                   norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                                   conv_cfg=dict(type='Conv2d', bias=False)),
                               pts_neck=dict(
                                   type='SECONDFPN',
                                   in_channels=[128, 256],
                                   out_channels=[256, 256],
                                   upsample_strides=[1, 2],
                                   norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                                   upsample_cfg=dict(type='deconv', bias=False),
                                   use_conv_for_no_stride=True),
                               pts_bbox_head=dict(
                                   type='CenterHead',
                                   in_channels=sum([256, 256]),
                                   tasks=[
                                       dict(num_class=1, class_names=['car']),
                                       dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                                       dict(num_class=2, class_names=['bus', 'trailer']),
                                       dict(num_class=1, class_names=['barrier']),
                                       dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                                       dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                                   ],
                                   common_heads=dict(
                                       reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
                                   share_conv_channel=64,
                                   bbox_coder=bbox_coder,
                                   train_cfg=train_cfg,
                                   test_cfg=test_cfg,
                                   separate_head=dict(
                                       type='SeparateHead', init_bias=-2.19, final_kernel=3),
                                   loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                                   loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
                                   norm_bbox=True),
                               )
        #############################################################################################

        self.labelenc_conf = dict(
            box_features=9,
            label_features=10,
            hidden_features=256,
            out_features=[128, 256],
            stride=[1, 2],
            feature_size=128
        )

        "centerpoint checkpoint"
        lidar_ckpt_path = './ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth'


        #############################################################################################
        #turn off bda
        self.bda_aug_conf_val = {
                                'rot_lim': (0, 0),
                                'scale_lim': (1, 1),
                                'flip_dx_ratio': 0.0,
                                'flip_dy_ratio': 0.0
                                }
        #############################################################################################
        self.model = LabelEncoder(self.head_conf,
                                  self.labelenc_conf,
                                  self.lidar_conf,
                                  lidar_ckpt_path=lidar_ckpt_path)


    def training_step(self, batch):
        (_, _, _, _, gt_boxes, gt_labels, depth_labels) = batch
        if torch.cuda.is_available():
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            bev_mask, bev_box, bev_label, targets = self.model.module.get_targets(gt_boxes, gt_labels)
        else:
            bev_mask, bev_box, bev_label, targets = self.model.get_targets(gt_boxes, gt_labels)

        preds = self.model(bev_mask, bev_box, bev_label)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            detection_loss = self.model.module.loss(targets, preds)
        else:
            detection_loss = self.model.loss(targets, preds)

        self.log('detection_loss', detection_loss)
        return detection_loss

    def eval_step(self, batch, batch_idx, prefix: str):
        (_, _, _, img_metas, gt_boxes, gt_labels) = batch

        if torch.cuda.is_available():

            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            bev_mask, bev_box, bev_label, targets = self.model.module.get_targets(gt_boxes, gt_labels)
        else:
            bev_mask, bev_box, bev_label, targets = self.model.get_targets(gt_boxes, gt_labels)
        preds = self.model(bev_mask, bev_box, bev_label)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)

        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def configure_optimizers(self):
        lr = 1e-3
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [10, 11])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(LabelDistillModel,
            'LabelDistill_step1',
            use_ema=True,
            extra_trainer_config_args={'epochs': 12}
            )
