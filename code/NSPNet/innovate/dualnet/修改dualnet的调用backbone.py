# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from stft_core.modeling import registry
from stft_core.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import resnet_dcn
#实现多个不同类型的骨干网络构建函数 并注册到全局注册表中 后续根据配置动态调用相应的构建函数

@registry.BACKBONES.register("ResDCN-18")
@registry.BACKBONES.register("ResDCN-34")
@registry.BACKBONES.register("ResDCN-50")
def build_resnetdcn_backbone(cfg):#将其注册为ResDCN-18   resnet_dcn构建主题网络部分
    body = resnet_dcn.get_center_net(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))#包装为一个nn.Sequential对象
    return model


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):#将其注册为R-50-C4到BACKBONES 方便需要时调用    resnet构建主体网络部分
    # body = resnet.ResNet(cfg)#使用resnet网络  根据传入类的配置cfg初始化并构建resnet网络结构
    body = resnet.DualNet(cfg)#使用resnet网络  根据传入类的配置cfg初始化并构建resnet网络结构
    model = nn.Sequential(OrderedDict([("body", body)]))#包装为序列化的pytorch模块,  将ResNet的forward作为body给到model
    # (OrderedDict([("body", body)])创建有序字典 值为前面构建resnet主体body
    # nn.Sequential是pytorch的一个模块容器,允许我们将多个模块按顺序组合在一起,这里只是一个简单的包含body的模型
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS#配置通道数 定义网络的输出特征图的通道数
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):#将其注册为R-50-FPN    resnet主体
    body = resnet.DualNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(#构建fpn 特征金字塔网络部分
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))#将俩者合并为一个nn.Sequential对象   这里是安装body  fpn的顺序组合为复杂的模型
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):#构建带有特征金字塔网络fpn 和 retinanet的resnet骨干网络
    body = resnet.DualNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels #1024 for retinanet, 256 for fcos
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):#该函数根据配置动态调用已注册的骨干网络构建函数  CONV_BODY指定具体的骨干网络R-50-C4 并从注册表获取相应的构建函数
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY#使用断言判断是否存在
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
