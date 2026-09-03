import argparse
import inspect
import torch.nn.functional as F

from improved_diffusion.LSCT import LSCClassifier, LSCClassifier_Visulation
import torch


# 用于创建LSCT的默认参数和创建函数
def model_defaults():
    return dict(
        image_size=256,
        patch_size=16,
        num_classes=5,
        dim=768,
        depth=12,
        heads=12,
        pool="mean",
        channels=1,
        dim_heads = 768 // 12,
        dropout=0.0,
        emb_dropout=0.0,
    )

def create_model(
    image_size,
    patch_size, 
    num_classes,
    dim,
    depth,
    heads,
    pool,
    channels,
    dim_heads,
    dropout,
    emb_dropout,
):
    return LSCClassifier(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        pool=pool,
        channels=channels,
        dim_heads=dim_heads,
        dropout=dropout,
        emb_dropout=emb_dropout,
    )

def create_model_visulation(
    image_size,
    patch_size, 
    num_classes,
    dim,
    depth,
    heads,
    pool,
    channels,
    dim_heads,
    dropout,
    emb_dropout,
):
    return LSCClassifier_Visulation(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        pool=pool,
        channels=channels,
        dim_heads=dim_heads,
        dropout=dropout,
        emb_dropout=emb_dropout,
    )

def create_trianing_loss(type):
    assert type in {"crossEntropy", "crossEntropyAndAcc"}
    if type == "crossEntropy":
        return compute_crossEntropy_loss
    elif type == "crossEntropyAndAcc":
        return compute_crossEntropy_and_acc

def compute_crossEntropy_loss(model, input, label):
    term = {}
    output, feat = model(input)
    loss = F.cross_entropy(output, label)
    term["loss"] = loss
    return term
def compute_crossEntropy_and_acc(model, input, label):
    term = {}
    output, feat = model(input)
    loss = F.cross_entropy(output, label)
    term["loss"] = loss
    with torch.no_grad():
        out = output.detach().cpu()
        gt = label.detach().cpu()
        max_value = torch.max(out)
        pred = torch.zeros_like(out, requires_grad=False)
        pred[max_value == out] = 1
        correct_num = torch.sum(pred == gt)
        acc = correct_num / (gt.shape[0] * gt.shape[1])
        term["acc"] = acc
    return term
    

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
