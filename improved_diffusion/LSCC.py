import torch.nn as nn
from improved_diffusion.CDM import *
from improved_diffusion.LSCT import *
from improved_diffusion.ViTC import *
from improved_diffusion.ResNetC import *

class LowRankSparseCompressControlNet(nn.Module):
    def __init__(self, LSCClassifier_type, num_doses, *args, **kwargs):
        super().__init__()
        assert LSCClassifier_type in {"LSCClassifier_Small", "LSCClassifier_Base", "ViTClassifier_Base_p16", "ResNetClassifier_base16"}
        if LSCClassifier_type == "LSCClassifier_Small":
            self.classifier = LSCClassifier_Small(num_doses)
        elif LSCClassifier_type == "LSCClassifier_Base":
            self.classifier = LSCClassifier_Base(num_doses)
        elif LSCClassifier_type == "ViTClassifier_Base_p16":
            self.classifier = vit_base_patch16(img_size=256, in_chans=1, num_classes=num_doses)
        elif LSCClassifier_type == "ResNetClassifier_base16":
            self.classifier = resnet_base16(num_classes=num_doses)
        self.controledIddpm = ControlIddpm(*args, **kwargs)
        # 关闭分类器的训练
        self.classifier = self.classifier.eval()
        self.classifier.train = disabled_train
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.params_control = self.controledIddpm.params_control

    def forward(self, x, timesteps, y=None, hint=None):
        with torch.no_grad():
            _, LSCV = self.classifier(hint)
        out = self.controledIddpm(x=x, timesteps=timesteps, y=y, hint=hint, LSCV=LSCV)
        return out


# if __name__ == "__main__":
#     model = LowRankSparseCompressControlNet(
#         LSCClassifier_type="LSCClassifier_Base", num_doses=5, in_channels=1, 
#         model_channels=128, out_channels=1, num_res_blocks=2,
#         attention_resolutions=(32,16,8), LSCV_dim=768, insert_LSCV=True,
#     )
#     x = torch.randn((2, 1, 256, 256))
#     t = torch.randn((2, ))
#     y = None
#     hint = torch.randn((2, 1, 256, 256))
#     out = model(x=x, timesteps=t, y=y, hint=hint)
#     pass