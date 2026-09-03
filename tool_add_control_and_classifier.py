import os
import torch
import argparse

from improved_diffusion.script_util import add_dict_to_argparser, args_to_dict
from improved_diffusion.script_util import create_LSCC_and_diffusion, LSCC_and_diffusion_defaults, NoneLSCC_and_diffusion_defaults

# 从name中去掉parent_name, 如果name中存在parent_name的话，返回True
def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ""
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ""
    return True, name[len(parent_name):]

# 从control_and_diffusion_defaults中获取默认参数
# 如果要换模型的话，就要替换defaults
def create_argparser():
    defaults = dict(
        classifier_path = "",
        diffusion_path = "",
        output_path = "",
    )
    defaults.update(LSCC_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

args = create_argparser().parse_args()

# input_path: 基础模型的预训练权重路径
# output_path: 输出添加控制权重以后的权重路径
classifier_path = args.classifier_path
diffusion_path = args.diffusion_path
output_path = args.output_path
assert os.path.exists(classifier_path), "Classifier model dose not exist!"
assert os.path.exists(diffusion_path), "Diffusion model dose not exist!"
assert not os.path.exists(output_path), "Output filename already exists!"
assert os.path.exists(os.path.dirname(output_path)), "Output path is not valid!"

# 用control_and_diffusion_defaults创建模型
# 如果要换模型的话，要换create和defaults
# 注意这里的模型是模型和控制模型结合的模型 from improved_diffusion.cdm
model, _ = create_LSCC_and_diffusion(
    **args_to_dict(args, LSCC_and_diffusion_defaults().keys())
)

# scratch_dict = model.state_dict()
# for k in scratch_dict.keys():
#     print(k)
# layer name of model examples:
# 1. classifier.transformer.layers.9.1.norm.bias
# 2. controledIddpm.controled_model.input_blocks.1.0.out_layers.3.weight
# 3. controledIddpm.control_model.input_blocks.10.0.in_layers.2.bias

scratch_dict = model.state_dict()
target_dict = {}
# 导入LSC分类器权重
classifier_pretrained_weights = torch.load(classifier_path, map_location=torch.device('cpu'))
if 'state_dict' in classifier_pretrained_weights:
    classifier_pretrained_weights = classifier_pretrained_weights['state_dict']
# 取出模型中每一层的名字，如果他是classifier.transformer开头，说明来自LSCClassifier
# 如果属于LSCClassifier，那么就换掉前缀，换成transformer
# 换完方便classifier_pretrained_weights的导入
for k in scratch_dict.keys():
    is_classifier, name = get_node_name(k, "classifier.transformer")
    if is_classifier:
        replaced_k = "transformer" + name
    else:
        replaced_k = None
    if replaced_k is not None and replaced_k in classifier_pretrained_weights:
        target_dict[k] = classifier_pretrained_weights[replaced_k].clone()
        print("load one for classifier")
    else:
        target_dict[k] = scratch_dict[k].clone()
# 导入diffusion权重
diffusion_pretrained_weights = torch.load(diffusion_path, map_location=torch.device('cpu'))
if 'state_dict' in diffusion_pretrained_weights:
    diffusion_pretrained_weights = diffusion_pretrained_weights['state_dict']
# 取出模型中每一层的名字，如果他是controledIddpm.control_model.的名字开头，说明来自controlNet
# 如果它属于controlNet，那就换掉前缀，换成""(基础模型的前缀)
# 如果他是controledIddpm.controled_model的名字开头，说明来自iddpm
# 如果它属于iddpm，那就换掉前缀，换成"", 方便权重的导入
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, "controledIddpm.control_model.")
    if is_control:
        copy_k = name
    else:
        is_iddpm, name = get_node_name(k, "controledIddpm.controled_model.")
        if is_iddpm:
            copy_k = name
        else:
            copy_k = None
    if copy_k is not None and copy_k in diffusion_pretrained_weights:
        target_dict[k] = diffusion_pretrained_weights[copy_k].clone()
        print("load one for diffusion and control")
    else:
        target_dict[k] = scratch_dict[k].clone()

# 保存出权重，要训练controlNet的时候使用resume的方式
model.load_state_dict(target_dict, strict=True)
print("load weights wall done!")
torch.save(model.state_dict(), output_path)
print('Done.')
