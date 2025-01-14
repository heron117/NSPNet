# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
import pywt

from stft_core.layers import FrozenBatchNorm2d  # STEM_FUNC中有使用
from stft_core.layers import Conv2d
from stft_core.layers import DFConv2d  # 可变形卷积
from stft_core.modeling.make_layers import group_norm
from stft_core.utils.registry import Registry
from stft_core.layers import ContextBlock  # 上下文块

# 定义了一个支持配置化的resnet模块,主要分为 配置类 resnet架构定义 辅助类 注册表
# 通过配置文件和注册表机制，动态地构建不同变体的 ResNet 模型。它定义了基础层、残差块、FPN 结构等，并支持多种归一化方式和可变形卷积

# ResNet stage specification
StageSpec = namedtuple(
    # StageSpec 是一个 namedtuple，用于定义 ResNet 的每个 stage 的属性，
    # 包括 index（阶段索引）、block_count（残差块数量）和 return_features（是否返回该阶段的特征）。
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages) asuvid本处使用的是这个
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)  # 定义了多个resnet和fpn特征金字塔的阶段规格


# 小波变换网络，用于处理高频和低频分量
# class WaveletNet(nn.Module):
#     def __init__(self):
#         super(WaveletNet, self).__init__()
#         # Define small convolutional networks for high and low frequency feature extraction
#         self.high_freq_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.low_freq_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         # Apply wavelet transform to get high and low frequency components
#         coeffs2 = pywt.dwt2(x, 'haar')  # Perform wavelet transform
#         LL, (LH, HL, HH) = coeffs2
#         # Process high and low frequency components
#         high_freq = self.high_freq_conv(torch.cat([LH, HL, HH], dim=1))
#         low_freq = self.low_freq_conv(LL)
#         return high_freq, low_freq

class WaveletNet(nn.Module):
    def __init__(self):
        super(WaveletNet, self).__init__()
        # Define small convolutional networks for high and low frequency feature extraction
        self.high_freq_conv = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.low_freq_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def forward(self, x):
        # 如果输入是在 GPU 上，我们需要将其移动到 CPU 以便进行小波变换
        if x.is_cuda:
            x_cpu = x.cpu().detach().numpy()  # 移动到 CPU 并转换为 NumPy 数组
        else:
            x_cpu = x.detach().numpy()  # 如果不是在 GPU 上，直接转换为 NumPy 数组

        # Apply wavelet transform to get high and low frequency components using PyWavelets
        coeffs2 = pywt.dwt2(x_cpu, 'haar')  # Perform wavelet transform
        LL, (LH, HL, HH) = coeffs2

        # 将小波变换的结果转回成 PyTorch 张量，并移回 GPU（如果原输入在 GPU 上）
        LL = torch.tensor(LL).to(x.device)
        LH = torch.tensor(LH).to(x.device)
        HL = torch.tensor(HL).to(x.device)
        HH = torch.tensor(HH).to(x.device)

        # Process high and low frequency components
        high_freq = self.high_freq_conv(torch.cat([LH, HL, HH], dim=1))
        low_freq = self.low_freq_conv(LL)

        return high_freq, low_freq

#selayer是一种通道注意力机制,它通过全局池化操作捕捉每个通道的重要性,然后通过一个全连接网络来生成每个通道的权重
#这些权重将决定每个通道的贡献大小,并通过逐通道相乘的方式对输入特征进行加权。

#输入通道数通过全局池化和全连接层降维,然后调整为输出通道数,然后用sigmoid激活来生成通道注意力权重
class SELayer(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, out_channels, bias=False),  # 输出通道调整为 out_channels
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, -1, 1, 1)  # 输出通道数调整
        return x * y.expand_as(x)  # 逐元素相乘，应用 SE 注意力

# 修改后的ResNet + WaveletNet模型
class DualNet(nn.Module):
    def __init__(self, cfg):
        super(DualNet, self).__init__()
        # 使用现有的ResNet并加载预训练权重
        self.resnet = ResNet(cfg)

        # 定义小波网络
        self.wavelet_net = WaveletNet()

        # 定义一个字典存储动态的 SE 注意力模块
        self.se_layers = {}

    def forward(self, x):
        outputs = []
        wavelet_high, wavelet_low = self.wavelet_net(x)  # 通过小波变换网络处理图像的高低频分量

        # 通过ResNet提取特征
        x = self.resnet.stem(x)

        for stage_name in self.resnet.stages:
            x = getattr(self.resnet, stage_name)(x)

            # 只在第二层（layer2）进行融合操作
            if stage_name == 'layer2':
                print(f"当前融合层 {stage_name}")

                # 调整大小确保 wavelet 特征与 ResNet 输出的大小匹配
                wavelet_high = F.interpolate(wavelet_high, size=x.shape[2:], mode='bilinear', align_corners=False)
                wavelet_low = F.interpolate(wavelet_low, size=x.shape[2:], mode='bilinear', align_corners=False)

                # 动态调整 SE 注意力层
                fused_feature = torch.cat([x, wavelet_high, wavelet_low], dim=1)  # 640通道
                fused_channels = fused_feature.shape[1]
                output_channels = x.shape[1]  # ResNet第2层的输出通道为512
                print(f"fused_feature的形状: {fused_feature.shape}")

                # 动态生成 SE 注意力模块，如果还未创建或通道数不匹配
                if stage_name not in self.se_layers or self.se_layers[stage_name].fc[0].in_channels != fused_channels:
                    self.se_layers[stage_name] = SELayer(fused_channels, output_channels).to(fused_feature.device)

                # SE 注意力操作，注意输出通道调整为512
                fused_feature = self.se_layers[stage_name](fused_feature)
                print(f"outputs_feature的形状: {fused_feature.shape}")
                outputs.append(fused_feature)
            elif self.resnet.return_features[stage_name]:
                # 在其他层返回特征，但不进行小波融合
                outputs.append(x)

        return outputs

class ResNet(nn.Module):
    def __init__(self, cfg):  # 根据配置文件构建 ResNet 模型，包括 stem 模块和各个阶段的模块。配置中指定了各个模块的类型、层数、是否返回特征图等
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]  # StemWithFixedBatchNorm-即为BaseStem的forward
        stage_specs = _STAGE_SPECS[
            cfg.MODEL.BACKBONE.CONV_BODY]  # R-50-C4   R-50-FPN-RETINANET-yaml中是这个---即为ResNet50FPNStagesTo5
        # StageSpec(index=i, block_count=c, return_features=r) for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]  # BottleneckWithFixedBatchNorm

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS  # _C.MODEL.RESNETS.NUM_GROUPS = 1
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP  # _C.MODEL.RESNETS.WIDTH_PER_GROUP = 64
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS  # _C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
        stage2_bottleneck_channels = num_groups * width_per_group  # 64
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  # _C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:  # 定义并构建resnet模型的各个阶段  遍历元组stage_spec,stage_specs内部包含了模型各个阶段的信息 块数和是否返回特征图
            name = "layer" + str(
                stage_spec.index)  # 设置层名称  index的值来自于stage_spec.index 是一个包含了模型阶段信息的对象,index由元组stage_spec的定义决定属性tuple。stage_specs中已写好
            # 结果是layer1 layer2 layer3 取决于index的值
            stage2_relative_factor = 2 ** (stage_spec.index - 1)  # 计算通道大小   随阶段索引成倍增加
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor  # 计算当前阶段的瓶颈通道数
            out_channels = stage2_out_channels * stage2_relative_factor  # 计算当前阶段的输出通道数
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[
                stage_spec.index - 1]  # 可变形卷积网络dcn STAGE_WITH_DCN: (False, True, True, True)确定当前阶段是否使用 不同index进行不同的使用
            stage_with_gcb = cfg.MODEL.RESNETS.STAGE_WITH_GCB[
                stage_spec.index - 1]  # 全局上下文模块gcb STAGE_WITH_GCB: (False, True, True, True)-表示在234阶段启用gcb模块
            # [stage_spec.index -1]是用于一个以0为起点的索引列表中访问相应的配置,resnet的不同阶段按照顺序进行编号,index1 2 3,由于python的列表是从0开始索引,所以需要减去1才能正确的访问配置列表中的元素。
            # index的值为不同的值 指向配置列表中的不同的元素
            gcb = cfg.MODEL.RESNETS.GCB if stage_with_gcb else None  # 如果阶段使用gcb 就赋值给gcb
            module = _make_stage(  # 创建阶段模块 使用指定的 转换模块 输入和输出通道  块数  和dcn gcb配置
                transformation_module,
                in_channels,
                bottleneck_channels,  # 瓶颈层的通道数
                out_channels,
                stage_spec.block_count,  # 残差块数
                num_groups,  # 组规范化 或分组卷积的组数
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,  # 是否使用步幅
                first_stride=int(stage_spec.index > 1) + 1,  # 阶段中第一个卷积层的步幅
                dcn_config={  # 可变形卷积的配置
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,  # WITH_MODULATED_DCN: False
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,  # DEFORMABLE_GROUPS: 1
                },
                gcb=gcb
            )
            in_channels = out_channels  # 更新下一个阶段的输入通道数以匹配当前阶段的输出通道数
            self.add_module(name,
                            module)  # 将构建的阶段模块添加到网络中   add_module是pytorch的一个方法 用于向当前模块中添加子模块,name是子模块的名称,module是子模块实例
            # _make_stage函数会根据stage_spec和配置文件中的其他参数创建一个阶段模块module
            self.stages.append(name)  # 使用append方法将阶段名称添加到阶段列表中 用于存储阶段的名称,在forward中访问self.stages
            self.return_features[name] = stage_spec.return_features  # 注册该阶段是否返回其特征

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)  # default fix stem and stage1    值为2
        # 会冻结stem和第一阶段
        # 获得了骨干网络的各个模块层级 但是如何修改了权重和进行推理时的调用呢?自己还是没有找到

    def _freeze_backbone(self, freeze_at):  # 根据配置冻结部分参数，使其在训练过程中不更新
        if freeze_at < 0:  # 如果值小于0 不冻结任何部分 直接返回
            return
        for stage_index in range(freeze_at):  # 遍历从0到值,依次冻结对应的层
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem  对于
            else:
                m = getattr(self, "layer" + str(stage_index))  # 使用getattr冻结对应的层
                # getattr(object,name) 用于从对象object中,获取名为name的属性--可以灵活动态的访问属性,而无需明确的写出属性名
            for p in m.parameters():
                p.requires_grad = False  # 将其设置为false 防止这些部分在训练过程中被更新

    def forward(self, x):  # 定义前向传播，依次通过 stem 和各个阶段的模块，输出需要的特征图
        outputs = []  # 存储需要返回的特征图
        x = self.stem(x)  # 经历 _STEM_MODULES 进行处理
        for stage_name in self.stages:  # 遍历列表,依次通过每个阶段的模块进行处理
            x = getattr(self, stage_name)(x)  # 获取对应的层 并对输入输入进行处理
            if self.return_features[stage_name]:  # 如果返回为true 则将该特征图添加到outputs列表中
                outputs.append(x)
        return outputs
    # 在初始化resnet模型时,stem_module是在所有阶段模块之前构建的,并且作为模型的一个成员变量self.stem存在,
    # 具体来说forward方法中,输入x会先通过self.stem处理,然后再通过各个阶段模块处理


class ResNetHead(nn.Module):  # ResNetHead：定义了 ResNet 的头部，用于从特征图中提取高层次的特征。
    def __init__(
            self,
            block_module,  # 表示用于构建块的模块类型
            stages,  # 表示网络中的不同阶段
            num_groups=1,  # 配置网络的细节
            width_per_group=64,
            stride_in_1x1=True,
            stride_init=None,
            res2_out_channels=256,
            dilation=1,
            dcn_config={},
            gcb=None
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)  # 根据阶段索引计算相对因子，用于调整通道数
        stage2_bottleneck_channels = num_groups * width_per_group  # 计算每个阶段的瓶颈通道数和输出通道数。
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2  # 初始化输入通道数和瓶颈通道数。
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]  # 将字符串名称的模块转换为实际的模块类

        self.stages = []  # 用于存储阶段名称
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)  # 循环遍历每个阶段，根据其索引设置名称和步长
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(  # 用 _make_stage 函数构建每个阶段的模块，并将其添加到当前模块中   _make_stage函数很有意思
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config,
                gcb=gcb
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels  # 设置输出通道

    def forward(self, x):  # 定义前向传播的过程
        for stage in self.stages:  # 循环遍历列表self.stages的每个阶段 ,依次调用每个阶段的模块处理输入x
            x = getattr(self, stage)(x)  # 获取当前阶段的模块并对x进行处理
        return x  # 最终返回处理后的x


def _make_stage(  # _make_stage：辅助函数，用于构建 ResNet 的每个 stage，包括多个残差块。构造为module
        transformation_module,
        in_channels,
        bottleneck_channels,
        out_channels,
        block_count,
        num_groups,
        stride_in_1x1,
        first_stride,
        dilation=1,
        dcn_config={},
        gcb=None
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):  # 循环blockcount次数,创建每个块并添加到blocks列表中
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config,
                gcb=gcb
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)  # 将所有块组合为一个顺序容器并返回


class Bottleneck(nn.Module):  # 定义了一个残差块 包括1*1 3*3 1*1 的卷积块
    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups,
            stride_in_1x1,
            stride,
            dilation,
            norm_func,
            dcn_config,
            gcb
    ):
        super(Bottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:  # 如果in_channels 不等于 out_channels 则使用下采样层来使其匹配
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample, ]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1  # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(  # 使用了可变形卷积（Deformable Convolutional Networks）
                bottleneck_channels,
                bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn,
                kernel_size=3,
                stride=stride_3x3,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        self.with_gcb = gcb is not None
        if self.with_gcb:
            self.context_block = ContextBlock(inplanes=out_channels, ratio=gcb)
            # 上下文块（Context Block），根据配置选择性地添加这些模块

        for l in [self.conv1, self.conv3, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.with_gcb:
            out = self.context_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out


class BaseStem(nn.Module):  # 基础层类 定义了resnet的基础层,包括一个7*7的卷积层,批标准化层 和 最大池化层
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)  # norm_func=FrozenBatchNorm2d批标准化    初始化时使用,调用时直接使用的是forward

        for l in [self.conv1, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):  # StemWithFixedBatchNorm调用时直接使用的是forward-即stem_module所调用的部分
        x = self.conv1(x)  # 进行卷积
        x = self.bn1(x)  # 进行批标准化
        x = F.relu_(x)  # 进行激活
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)  # 进行降维池化
        return x


class BottleneckWithFixedBatchNorm(
    Bottleneck):  # BottleneckWithFixedBatchNorm 类继承自 Bottleneck，只不过使用了固定批标准化层 FrozenBatchNorm2d。批标准化类
    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups=1,
            stride_in_1x1=True,
            stride=1,
            dilation=1,
            dcn_config={},
            gcb=None
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config,
            gcb=gcb
        )


class StemWithFixedBatchNorm(BaseStem):  # StemWithFixedBatchNorm 类继承自 BaseStem，也使用了固定批标准化层 FrozenBatchNorm2d。
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(  # 调用父类的方法  并传递cfg 和norm_func=FrozenBatchNorm2d作为参数
            cfg, norm_func=FrozenBatchNorm2d  # 使得该类可以使用该函数进行批标准化
        )


class BottleneckWithGN(Bottleneck):  # 组归一化类 BottleneckWithGN 类继承自 Bottleneck，使用了组归一化层 group_norm。
    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups=1,
            stride_in_1x1=True,
            stride=1,
            dilation=1,
            dcn_config={}
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )


class StemWithGN(BaseStem):  # StemWithGN 类继承自 BaseStem，使用了组归一化层 group_norm
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,  # 本处使用的是这个
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})
# _TRANSFORMATION_MODULES 注册表包含了不同类型的瓶颈模块。_STEM_MODULES 注册表包含了不同类型的 stem 模块。_STAGE_SPECS 注册表包含了不同类型的 ResNet 阶段规格。
