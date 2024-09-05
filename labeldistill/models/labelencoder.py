from torch import nn

from labeldistill.layers.backbones.label_backbone import LabelBackbone

from labeldistill.layers.heads.kd_head import KDHead
from mmdet3d.models import build_detector
import torch

__all__ = ['LabelEncoder']


class LabelEncoder(nn.Module):
    def __init__(self,
                 head_conf,
                 labelenc_conf,
                 lidar_conf=None,
                 lidar_ckpt_path=None):
        super(LabelEncoder, self).__init__()
        self.head = KDHead(**head_conf)
        self.label_encoder = LabelBackbone(**labelenc_conf)

        self.centerpoint = build_detector(lidar_conf)

        lidar_params = torch.load(lidar_ckpt_path, map_location='cpu')

        prefix = 'model.centerpoint.'
        load_keys = [k for k in lidar_params['state_dict'] if k.startswith(prefix)]
        self.centerpoint.load_state_dict({k[len(prefix):]: lidar_params['state_dict'][k] for k in load_keys})
        self.centerpoint.eval()

        for param in self.centerpoint.parameters():
            param.requires_grad = False


    def forward(
        self,
        bev_mask=None,
        bev_box=None,
        bev_label=None,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """

        label_feats = self.label_encoder(bev_box, bev_label, bev_mask.unsqueeze(1))
        label_feats = self.centerpoint.pts_neck(label_feats)
        label_pred = self.centerpoint.pts_bbox_head(label_feats)

        return label_pred


    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
