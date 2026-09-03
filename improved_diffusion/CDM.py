import torch
import torch.nn as nn
import torch.nn.init as init
from improved_diffusion.unet import *

def zero_module_(module):
    init.zeros_(module.weight)
    # init.normal_(module.weight, mean = 0, std = 0.01)
    # module.weight.data = module.weight.data + 1
    return module

# 用于被控制的基础Unet，继承自UNetModel
# 覆写forward，在输入项中加入control，这个control是一个controlNet返回的列表
# 这个列表是编码器的中间特征图的输出
# forward中逐个pop controlNet的列表，然后拼在UNet解码器的中间特征上
class ControledUnetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # type: ignore

    def forward(self, x, timesteps, y=None, control=None, only_mid_control=False, **kwargs):
        # param: x 输入带噪声图像
        # param: timesteps 时间步
        # param: y 类别标签（几乎是用不到）
        # param: control 控制条件，需要的是控制网络输出的一个out列表
        # param: only_mid_control 是否只控制中间层
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        with torch.no_grad():
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
            if self.num_classes is not None:
                assert y.shape == (x.shape[0],)
                emb = emb + self.label_emb(y)
            
            h = x.type(self.inner_dtype)
            for module in self.input_blocks:
                h = module(h, emb)
                hs.append(h)
            h = self.middle_block(h, emb)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            # if only_mid_control or control is None:
            if control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)

# 用于控制的编码器头
# 这个结构需要和基础Unet的基础头和Unet的最底层需要一模一样
# 所以如果基础Unet模型有改变的话，这个也需要修改  
class ControlNet(nn.Module):
    def __init__(self,
                 in_channels,
                 model_channels,
                 hint_channels,
                 num_res_blocks,
                 attention_resolutions,
                 dropout=0.0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=2,
                 num_classes=None,
                 use_checkpoint=False,
                 num_heads=1,
                 num_heads_upsample=-1,
                 use_scale_shift_norm=False,                 
                 ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.hint_channels = hint_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims
        
        self.time_embed_dim = time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_hint_block = TimestepEmbedSequential(
            zero_module_(conv_nd(dims, hint_channels, model_channels, 3, padding=1))
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        self.input_blocks = nn.ModuleList([
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ])
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
        
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
            return TimestepEmbedSequential(zero_module_(conv_nd(self.dims, channels, channels, 1, padding=0)))
    
    def get_time_embed_dim(self):
        return self.time_embed_dim
    
    def forward(self, x, timesteps, y=None, hint=None, **kwargs):
        # param: x: 输入带噪声图
        # param: timesteps: 时间步
        # param: y: 类别标签（几乎用不到）
        # param: hint: 控制图像的输入
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        guided_hint = self.input_hint_block(hint, emb)

        outs = []

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        h = x.type(self.inner_dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb)
            outs.append(zero_conv(h, emb))

        h = self.middle_block(h, emb)
        outs.append(self.middle_block_out(h, emb))

        return outs
    
    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

# 添加低秩稀疏向量的ControlNet。继承自ControlNet，覆写forward修改输出的control
class ControlNetWithLSCT(ControlNet):
    def __init__(self, LSCV_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        time_embed_dim = self.get_time_embed_dim()
        self.LSCV_embed = nn.Sequential(
            linear(LSCV_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.zero_linear = self.make_zero_linear(time_embed_dim)
        
    def make_zero_linear(self, channels):
        return TimestepEmbedSequential(zero_module_(linear(channels, channels)))

    def forward(self, x, timesteps, LSCV, y=None, hint=None):
        # param: x: 输入带噪声图
        # param: timesteps: 时间步
        # param: LSCV: 低秩稀疏压缩向量 LowRank Sparse Compress Vector
        # param: y: 类别标签（几乎用不到）
        # param: hint: 控制图像的输入
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        emb_t = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb_LSC = self.zero_linear(self.LSCV_embed(LSCV), emb_t)
        # emb_LSC = self.LSCV_embed(LSCV)
        assert emb_t.shape == emb_LSC.shape
        emb = emb_t + emb_LSC

        guided_hint = self.input_hint_block(hint, emb)

        outs = []

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        h = x.type(self.inner_dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb)
            outs.append(zero_conv(h, emb))

        h = self.middle_block(h, emb)
        outs.append(self.middle_block_out(h, emb))

        return outs


# 综合基础的Unet和controlNet的模型
# 对比Unet加了两个参数，
# only_mid_control用于是否只需要在最底层添加控制
# sd_locked用于是否需要锁定基础Unet的权重
class ControlIddpm(nn.Module):
    def __init__(self,
                in_channels,
                model_channels,
                out_channels,
                num_res_blocks,
                attention_resolutions,
                LSCV_dim=768,
                dropout=0,
                channel_mult=(1, 1, 2, 2, 4, 4),
                conv_resample=True,
                dims=2,
                num_classes=None,
                use_checkpoint=False,
                num_heads=1,
                num_heads_upsample=-1,
                use_scale_shift_norm=False,
                only_mid_control=False,
                sd_locked=True,
                insert_LSCV=False,
                *args, **kwargs,
                ):
        super().__init__()
        self.only_mid_control = only_mid_control
        self.sd_locked = sd_locked
        self.insert_LSCV = insert_LSCV
        self.controled_model = ControledUnetModel(
            in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult,
            conv_resample, dims, num_classes, use_checkpoint, num_heads, num_heads_upsample, use_scale_shift_norm,
        )
        if sd_locked:
            self.controled_model = self.controled_model.eval()
            self.controled_model.train = disabled_train
            for param in self.controled_model.parameters():
                param.requires_grad = False
        if insert_LSCV:
            self.control_model = ControlNetWithLSCT(
                in_channels=in_channels, model_channels=model_channels, hint_channels=in_channels, num_res_blocks=num_res_blocks,
                attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, 
                dims=dims, num_classes=num_classes, LSCV_dim=LSCV_dim, use_checkpoint=use_checkpoint, num_heads=num_heads,
                num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            )
        else:
            self.control_model = ControlNet(
                in_channels=in_channels, model_channels=model_channels, hint_channels=in_channels, num_res_blocks=num_res_blocks,
                attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, num_classes=num_classes,
                use_checkpoint=use_checkpoint, num_heads=num_heads, num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            )
        self.params_control = self.parameters()
        # self.params_control = self.named_parameters()

    def forward(self, x, timesteps, y=None, hint=None, LSCV=None):
        # param: x: 输入带噪声图
        # param: timesteps: 时间步
        # param: hint: 输入控制图像
        # param: y: 类别标签（几乎用不到）
        if self.insert_LSCV:
            control_out = self.control_model(x=x, timesteps=timesteps, y=y, hint=hint, LSCV=LSCV)
        else:
            control_out = self.control_model(x=x, timesteps=timesteps, y=y, hint=hint)  
        out = self.controled_model(x=x, timesteps=timesteps, y=y, control=control_out, only_mid_control=self.control_model)
        return out
        

# 用disabled_train用来覆盖model.train，因此来禁止模型调用train
def disabled_train(self, mode=True):
    return self

# if __name__ == "__main__":
# model = ControlIddpm(
#         in_channels=1, model_channels=128, out_channels=1, num_res_blocks=2,
#         attention_resolutions=(32,16,8), LSCV_dim=768, insert_LSCV=True,
#     )
# x = torch.randn((2, 1, 256, 256))
# t = torch.randn((2, ))
# y = None
# hint = torch.randn((2, 1, 256, 256))
# LSCV = torch.randn((2, 768))
# out = model(x=x, timesteps=t, y=y, hint=hint, LSCV=LSCV)
# pass