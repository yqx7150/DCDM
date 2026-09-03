'''
make attention map on image 
'''
import numpy as np
import torch as th
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("..")

from tqdm import tqdm
from sklearn.metrics import f1_score
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from LSCT.script_util_ResNet import (
    model_defaults,
    create_model,
    args_to_dict,
    add_dict_to_argparser,
)
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2
import numpy as np

def main():
    args = create_argparser().parse_args()
    # if args.device == "cuda0":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # elif args.device == "cuda1":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    dist_util.setup_dist()
    logger.configure(dir="/home/lqg/Desktop/HR/LDPET/ResNetClassifier_sample")

    logger.log("creating CLASSIFIER model...")
    model = create_model(
        **args_to_dict(args, model_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=args.deterministic,
        dose=args.dose,
    )

    logger.log("predition...")
    totalNumber = 0.
    correctNumber = 0.
    f1 = 0.
    for step, batchNcond in tqdm(enumerate(data), desc="LSCClassifier testing..."):
        if step >= args.num_samples:
            break
        batch, cond = batchNcond
        batch = batch.to(dist_util.dev())
        img = cond['lq'].to(dist_util.dev())
        label = cond['dose'].to(dist_util.dev())
        # print(label)
        sp = "/home/lqg/HR/codes/LDPET/classifiers_feature_logger/resnetlabel.txt"
        if  sp is not None:
            # 将特征向量以文本形式保存到文件
            with open(sp, 'a') as f:
                feature = label.detach().cpu().numpy()
                feature_str = ' '.join(map(str, feature))
                f.write(feature_str)
        model.eval()

        pred = model(img)[0]
        predlabel = pred.detach()
        max_value, max_index = th.max(predlabel, dim=1, keepdim=True)
        predlabel = (predlabel == max_value).float()
        assert predlabel.shape == label.shape
        B, _ = label.shape
        for i in range(B):
            if th.equal(predlabel[i], label[i]):
                correctNumber += 1
            totalNumber += 1

        predlabel = predlabel.cpu().numpy().flatten()  
        label = label.detach().cpu().numpy().flatten()
        f1 += f1_score(predlabel, label)
    print("Top1: {:.2f}%".format(100 * correctNumber / totalNumber))
    print("F1 score: {:.2f}".format(f1/step))


def create_argparser():
    defaults = dict(
        data_dir="/home/lqg/HR/dataset/eval/PART2",
        model_path="",
        weight_decay=0.0,
        batch_size=32,
        image_size=256,
        microbatch=-1,  # -1 disables microbatches
        use_fp16=False,
        fp16_scale_growth=1e-3,
        deterministic=True,
        device="cuda0",
        dose="ALL",
        class_cond=True,
        num_samples=300,
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()