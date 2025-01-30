import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AvgPool2d((9,9))
        self.max_pool = nn.MaxPool2d((9,9))

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out =self.shared_MLP(self.max_pool(x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes,8)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x
        x = self.ca(x) * x
        rub1 = out-x
        x = self.sa(x) * x
        rub2 = out-x
        return x+out,rub1,rub2

class PEFN(nn.Module):
    def __init__(self, fc_cls):
        super(PEFN, self).__init__()
        resnet_50 = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            resnet_50.conv1,
            resnet_50.bn1,
            resnet_50.relu,
            resnet_50.maxpool,
            resnet_50.layer1,
            resnet_50.layer2,
            resnet_50.layer3
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512,stride=1, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1,stride=1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512), Bottleneck(2048, 512))
        self.layer4.load_state_dict(resnet_50.layer4.state_dict())
        self.p2_0 = Bottleneck(1024, 512,stride=1, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1,1, bias=False), nn.BatchNorm2d(2048)))
        self.p2_0.load_state_dict(resnet_50.layer4[0].state_dict())
        self.p2_1 = Bottleneck(2048, 512)
        self.p2_1.load_state_dict(resnet_50.layer4[1].state_dict())
        self.p2_2 = Bottleneck(2048, 512)
        self.p2_2.load_state_dict(resnet_50.layer4[2].state_dict())
        self.rub = ReductionFc(2048, 512, fc_cls)
        self.cbam = CBAM(2048)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(18, 18))
        self.avgpool1 = nn.AvgPool2d((18,18))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(16, 16))
        self.p1_g_rd = ReductionFc(2048, 512, fc_cls)

        self.p1_part_pool1 = nn.MaxPool2d((3, 16))
        self.p1_part_pool2 = nn.MaxPool2d((16, 6))
        self.parth1_rd = ReductionFc(2048, 512, fc_cls)
        self.parth2_rd = ReductionFc(2048, 512, fc_cls)
        self.parth3_rd = ReductionFc(2048, 512, fc_cls)
        self.parth4_rd = ReductionFc(2048, 512, fc_cls)
        self.parth5_rd = ReductionFc(2048, 512, fc_cls)
        self.parth6_rd = ReductionFc(2048, 512, fc_cls)
        self.partv1_rd = ReductionFc(2048, 512, fc_cls)
        self.partv2_rd = ReductionFc(2048, 512, fc_cls)
        self.partv3_rd = ReductionFc(2048, 512, fc_cls)

        self.p2_pool = nn.MaxPool2d((16, 16))
        self.p2_b1_rd = ReductionFc(2048, 512, fc_cls)
        self.p2_b2_rd = ReductionFc(2048, 512, fc_cls)
        self.p2_b3_rd = ReductionFc(2048, 512, fc_cls)

    def forward(self, x):
        x = self.backbone(x)
        p1 = self.layer4(x)

        p1_h1 = p1[:, :, 0:9, 0:9]
        p1_h2 = p1[:, :, 0:9, 9:18]
        p1_v1 = p1[:, :, 9:18, 0:9]
        p1_v2 = p1[:, :, 9:18, 9:18]

        p1_1,r11,r12 = self.cbam(p1_h1)
        p1_2,r21,r22 = self.cbam(p1_h2)
        p1_3,r31,r32 = self.cbam(p1_v1)
        p1_4,r41,r42 = self.cbam(p1_v2)

        p1h1 = torch.cat((p1_1, p1_2), dim=3)
        p1h2 = torch.cat((p1_3, p1_4), dim=3)
        p1_ = torch.cat((p1h1, p1h2), dim=2)
        res = p1 - p1_#unobtrusive


        r1_1 = torch.cat((r11, r21), dim=3)
        r1_2 = torch.cat((r31, r41), dim=3)
        r1 = torch.cat((r1_1, r1_2), dim=2)
        r2_1 = torch.cat((r12, r22), dim=3)
        r2_2 = torch.cat((r32, r42), dim=3)
        r2 = torch.cat((r2_1, r2_2), dim=2)


        p1_g = self.maxpool1(p1_)  # b,2048,1,1
        p1_g_tri, p1_g_fc = self.p1_g_rd(p1_g)
        p1_avg = self.avgpool1(p1_)
        p1_m_tri, p1_m_fc = self.p1_g_rd(p1_avg)


        p1_parth = self.p1_part_pool1(p1_)  # b,2048,6,1
        p1_parth1 = p1_parth[:, :, 0:1, :]
        p1_parth2 = p1_parth[:, :, 1:2, :]
        p1_parth3 = p1_parth[:, :, 2:3, :]
        p1_parth4 = p1_parth[:, :, 3:4, :]
        p1_parth5 = p1_parth[:, :, 4:5, :]
        p1_parth6 = p1_parth[:, :, 5:6, :]  # b,2048,1,1

        p1_partv = self.p1_part_pool2(p1_)  # b,2048,1,3
        p1_partv1 = p1_partv[:, :, :, 0:1]
        p1_partv2 = p1_partv[:, :, :, 1:2]
        p1_partv3 = p1_partv[:, :, :, 2:3]  # b,2048,1,1

        p1_parth1_tri, p1_parth1_fc = self.parth1_rd(p1_parth1)
        p1_parth2_tri, p1_parth2_fc = self.parth2_rd(p1_parth2)
        p1_parth3_tri, p1_parth3_fc = self.parth3_rd(p1_parth3)
        p1_parth4_tri, p1_parth4_fc = self.parth4_rd(p1_parth4)
        p1_parth5_tri, p1_parth5_fc = self.parth5_rd(p1_parth5)
        p1_parth6_tri, p1_parth6_fc = self.parth6_rd(p1_parth6)

        p1_partv1_tri, p1_partv1_fc = self.partv1_rd(p1_partv1)
        p1_partv2_tri, p1_partv2_fc = self.partv2_rd(p1_partv2)
        p1_partv3_tri, p1_partv3_fc = self.partv3_rd(p1_partv3)


        rub1 = torch.matmul(r1, r2.permute(0, 1, 3, 2))+res
        rub1 = nn.MaxPool2d(kernel_size=(16, 16))(rub1)#b,2048,1,1
        rub1_tri, _ = self.rub(rub1)


        p2_branch1 = self.p2_0(x)
        p2_branch2 = self.p2_1(p2_branch1)
        p2_branch3 = self.p2_2(p2_branch2)




        p2_01 = torch.matmul(p2_branch1, p2_branch2)  # b,2048,18,18 .permute(0,1,3,2)
        p2_02 = torch.matmul(p2_branch1, p2_branch3)
        p2_12 = torch.matmul(p2_branch2, p2_branch3)



        p2_01_pool = self.maxpool2(p2_01)  # b,2048,1,1
        p2_02_pool = self.maxpool2(p2_02)
        p2_12_pool = self.maxpool2(p2_12)

        p2_01_tri, p2_01_fc = self.p2_b1_rd(p2_01_pool)
        p2_02_tri, p2_02_fc = self.p2_b2_rd(p2_02_pool)
        p2_12_tri, p2_12_fc = self.p2_b3_rd(p2_12_pool)

        p2_tri = torch.cat([p2_01_tri, p2_02_tri, p2_12_tri], dim=1)


        predict = torch.cat(
            [p1_g_tri,p1_m_tri,p1_parth1_tri, p1_parth2_tri, p1_parth3_tri, p1_parth4_tri, p1_parth5_tri, p1_parth6_tri,
             p1_partv1_tri, p1_partv2_tri, p1_partv3_tri, p2_01_tri, p2_02_tri, p2_12_tri], dim=1)

        # cls = sum([p1_g_fc,p1_m_fc,p1_parth1_fc, p1_parth2_fc, p1_parth3_fc, p1_parth4_fc, p1_parth5_fc, p1_parth6_fc,
        #         p1_partv1_fc, p1_partv2_fc, p1_partv3_fc, p2_01_fc, p2_02_fc, p2_12_fc])
        return (p1_g_tri,p1_m_tri, p2_tri,rub1_tri), (
        p1_g_fc,p1_m_fc, p1_parth1_fc, p1_parth2_fc, p1_parth3_fc, p1_parth4_fc, p1_parth5_fc, p1_parth6_fc, p1_partv1_fc,
        p1_partv2_fc, p1_partv3_fc, p2_01_fc, p2_02_fc, p2_12_fc), predict
        # return cls
class ReductionFc(nn.Module):
    def __init__(self, feat_in, feat_out, num_classes):
        super(ReductionFc, self).__init__()

        self.reduction = nn.Sequential(nn.Conv2d(feat_in, feat_out, 1, bias=False),
                                       nn.BatchNorm2d(feat_out), nn.ReLU())
        self._init_reduction(self.reduction)
        self.fc = nn.Linear(feat_out, num_classes, bias=False)
        self._init_fc(self.fc)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        # nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        reduce = self.reduction(x).view(x.size(0), -1)
        fc = self.fc(reduce)
        return reduce, fc

