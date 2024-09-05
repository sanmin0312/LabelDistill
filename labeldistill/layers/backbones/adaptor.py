from torch import nn
from mmdet.models.backbones.resnet import BasicBlock


class DistillAdaptor(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 stride
                 ):
        super().__init__()

        blocks = []
        for i in range(len(in_features)):
            block = [
                nn.Conv2d(in_features[i],
                          out_features[i],
                          stride=stride[i],
                          kernel_size=3,
                          padding=1,
                          padding_mode='replicate')
            ]

            block.append(nn.BatchNorm2d(out_features[i]))
            block.append(nn.ReLU(inplace=True))
            block.append(BasicBlock(out_features[i], out_features[i]))
            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for i in range(len(self.blocks)):
            outs.append(self.blocks[i](x[i]))
        return outs