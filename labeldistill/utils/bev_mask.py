import numpy as np
import torch
from torch.nn.functional import one_hot
from skimage.draw import polygon


def gen_labelinput(bev_mask,
                   bev_box,
                   bev_label,
                   center,
                   width,
                   length,
                   label_box,
                   label_class):
    instance_bev_mask = np.zeros_like(bev_mask)

    mask_x = center[0]
    mask_y = center[1]
    mask_w = width * 0.5
    mask_l = length * 0.5
    yaw = label_box[6] - np.pi / 2
    W, H = bev_mask.shape

    # Compute the 4 corners of the bounding box in BEV coordinates
    corners = torch.tensor([[+mask_l / 2, +mask_w / 2],
                            [+mask_l / 2, -mask_w / 2],
                            [-mask_l / 2, -mask_w / 2],
                            [-mask_l / 2, +mask_w / 2]],
                            dtype=torch.float32,
                            device='cuda')
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    rot_mat = torch.tensor([[cos_yaw, -sin_yaw],
                            [sin_yaw, cos_yaw]],
                            dtype=torch.float32,
                            device='cuda')
    corners = torch.matmul(corners, rot_mat.transpose(1,0))

    corners = corners + torch.stack((mask_x, mask_y))

    # Draw a polygon for the bounding box on the mask
    corners = corners.cpu().numpy().astype(np.int32)

    #########################################################################################################
    inst_mask = np.asarray(polygon(corners[:,1], corners[:,0])).clip(0,W-1)
    bev_mask[inst_mask[0, :],inst_mask[1, :]] = 1

    # One hot encoding for class information
    label_class = one_hot(label_class, num_classes=10)

    # Fill labels into bev_label
    bev_label[inst_mask[0, :], inst_mask[1, :]] = label_class.type(torch.float32)
    bev_box[inst_mask[0, :], inst_mask[1, :]] = label_box
    #########################################################################################################

    return bev_mask, bev_box, bev_label


