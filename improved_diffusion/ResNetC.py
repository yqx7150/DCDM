import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet16(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet16, self).__init__()
        self.in_channels = 96
        self.conv1 = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 94, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 96*2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 96*4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 96*8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96*8 * BasicBlock.expansion, num_classes)
        self.save_path = "/home/lqg/HR/codes/LDPET/classifiers_feature_logger"
        

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        feat1 = F.avg_pool2d(out.detach(), kernel_size=(64, 64)).squeeze(-1).squeeze(-1)
        out = self.layer2(out)
        feat2 = F.avg_pool2d(out.detach(), kernel_size=(32, 32)).squeeze(-1).squeeze(-1)
        out = self.layer3(out)
        feat3 = F.avg_pool2d(out.detach(), kernel_size=(16, 16)).squeeze(-1).squeeze(-1)
        out = self.layer4(out)
        feat4 = F.avg_pool2d(out.detach(), kernel_size=(8, 8)).squeeze(-1).squeeze(-1)
        feats = [feat1, feat2, feat3, feat4]
        if self.save_path is not None:
            # 将特征向量以文本形式保存到文件
            for i, feat in enumerate(feats):
                save_path = f"{self.save_path}/ResNet out{i}.txt"
                with open(save_path, 'a') as f:
                    feature = feat.detach().cpu().numpy()
                    feature_str = ' '.join(map(str, feature))
                    f.write(feature_str)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # if self.save_path is not None:
        #     # 将特征向量以文本形式保存到文件
        #     with open(self.save_path, 'a') as f:
        #         feature = out.detach().cpu().numpy()
        #         feature_str = ' '.join(map(str, feature))
        #         f.write(feature_str)

        out = self.fc(out)        
        return out, out.detach()

def resnet_base16(num_classes):
    return ResNet16(num_classes=num_classes)

# if __name__ == "__main__":
#     model = ResNet16(num_classes=5)
#     x = torch.randn((1, 1, 256, 256))
#     y = model(x)
#     print(y.shape)