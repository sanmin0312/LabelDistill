# Copyright (c) Megvii Inc. All rights reserved.
"""

"""
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.base_exp import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.models.centerpoint import CenterPoint
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR
from bevdepth.datasets.nusc_det_dataset_lidar import NuscDetDataset, collate_fn
from functools import partial
import os

import torch
import torch.nn.functional as F
class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_return_lidar = True

        self.train_info_paths = os.path.join(self.data_root,
                                             'nuscenes_infos_train2.pkl')
        self.val_info_paths = os.path.join(self.data_root,
                                           'nuscenes_infos_val2.pkl')

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

        #############################################################################################
        self.model = CenterPoint(self.lidar_conf, self.head_conf)


    def training_step(self, batch):
        (_, _, _, _, gt_boxes, gt_labels, lidar_pts, depth_labels) = batch
        if torch.cuda.is_available():
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
            self.model = self.model.cuda()
        lidar_preds = self.model(lidar_pts)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, lidar_preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss  = self.model.loss(targets, lidar_preds)

        self.log('detection_loss', detection_loss)

        return detection_loss

    def eval_step(self, batch, batch_idx, prefix: str):
        (_, mats, _, img_metas, _, _, lidar_pts) = batch

        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
        preds = self.model(lidar_pts)
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

    def train_dataloader(self):
        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                       bda_aug_conf=self.bda_aug_conf,
                                       classes=self.class_names,
                                       data_root=self.data_root,
                                       info_paths=self.train_info_paths,
                                       is_train=True,
                                       use_cbgs=self.data_use_cbgs,
                                       img_conf=self.img_conf,
                                       num_sweeps=self.num_sweeps,
                                       sweep_idxes=self.sweep_idxes,
                                       key_idxes=self.key_idxes,
                                       return_depth=self.data_return_depth,
                                       return_lidar=self.data_return_lidar,
                                       use_fusion=self.use_fusion)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion,
                               is_return_lidar=self.data_return_lidar),
            sampler=None,
        )
        return train_loader


    def val_dataloader(self):
        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                       bda_aug_conf=self.bda_aug_conf,
                                       classes=self.class_names,
                                       data_root=self.data_root,
                                       info_paths=self.val_info_paths,
                                       is_train=False,
                                       img_conf=self.img_conf,
                                       num_sweeps=self.num_sweeps,
                                       sweep_idxes=self.sweep_idxes,
                                       key_idxes=self.key_idxes,
                                       return_depth=False,
                                       return_lidar=self.data_return_lidar,
                                       use_fusion=self.use_fusion)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=False,
                               is_return_lidar=self.data_return_lidar),
            sampler=None,
        )
        return train_loader


    def configure_optimizers(self):
        # lr = 1e-4
        # optimizer = torch.optim.AdamW(self.model.parameters(),
        #                               lr=lr,
        #                               weight_decay=1e-2)
        # scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=lr*10, cycle_momentum=False)
        # return [[optimizer], [scheduler]]

        lr = 1e-4
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-2)
        scheduler = MultiStepLR(optimizer, [16, 19])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'centerpoint_vox01_128x128_20e_10sweeps',
            use_ema=True,
            extra_trainer_config_args={'epochs': 20}
            )
