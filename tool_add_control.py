import os
import torch
import argparse

from improved_diffusion.script_util import add_dict_to_argparser, args_to_dict
from improved_diffusion.script_util import create_control_and_diffusion, control_and_diffusion_defaults

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
        input_path = "",
        output_path = "",
    )
    defaults.update(control_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

args = create_argparser().parse_args()

# input_path: 基础模型的预训练权重路径
# output_path: 输出添加控制权重以后的权重路径
input_path = args.input_path
output_path = args.output_path
assert os.path.exists(input_path), "Input model dose not exist!"
assert not os.path.exists(output_path), "Output filename already exists!"
assert os.path.exists(os.path.dirname(output_path)), "Output path is not valid!"

# 用control_and_diffusion_defaults创建模型
# 如果要换模型的话，要换create和defaults
# 注意这里的模型是模型和控制模型结合的模型 from improved_diffusion.cdm
model, _ = create_control_and_diffusion(
    **args_to_dict(args, control_and_diffusion_defaults().keys())
)

# 导入预训练权重
pretrained_weights = torch.load(input_path, map_location=torch.device('cpu'))
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
# 取出模型权重每一层的名字，如果他是control_名字开头，说明来自controlNet
# 如果它属于controlNet，那就换掉前缀，换成基础模型的前缀
# 换完前缀把权重导入
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, "control_")
    if is_control:
        copy_k = "controled_" + name
        print(f"There weights are added to control: {name}")
    else:
        copy_k = k 
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        # print(f"These weights are newly added: {k}")

# 保存出权重，要训练controlNet的时候使用resume的方式
model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
