
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



class CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction, out_channels)
        self.spatial_attention = SpatialAttention(kernel_size)
        # 初始化参数
        self._init_weights()

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out

    def _init_weights(self):
        # 初始化所有卷积层权重为 Kaiming 正态分布
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16, out_planes=None):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 如果未指定out_planes，则默认为in_planes
        out_planes = out_planes if out_planes is not None else in_planes
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, out_planes, 1, bias=False)  # 将输出通道数设为out_planes
        )
        self.sigmoid = nn.Sigmoid()
        # 3x3卷积，用于将 x 从 in_planes 转换为 out_planes
        self.channel_transform = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)


    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        # 先转换 x 的通道数再进行相乘
        x_transformed = self.channel_transform(x)
        sigmoid_out = self.sigmoid(out)  # 计算 sigmoid(out)

        # print(f"sigmoid_out的形状: {sigmoid_out.shape}")
        # # 打印 sigmoid(out) 的值
        # print(
        #     f"sigmoid(out) 的值 - Mean: {sigmoid_out.mean().item():.4f}, Std: {sigmoid_out.std().item():.4f}, Max: {sigmoid_out.max().item():.4f}, Min: {sigmoid_out.min().item():.4f}")
        # print(f"x_transformed的形状: {x_transformed.shape}")
        # # 打印 sigmoid(out) 的值
        # print(
        #     f"x_transformed的值 - Mean: {x_transformed.mean().item():.4f}, Std: {x_transformed.std().item():.4f}, Max: {x_transformed.max().item():.4f}, Min: {x_transformed.min().item():.4f}")

        return sigmoid_out * x_transformed  # 确保通道数匹配

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = self.sigmoid(self.conv1(concat))  # (B, 1, H, W)
        return attention * x  # 应用空间注意力

class WaveletNet(nn.Module):
    def __init__(self):
        super(WaveletNet, self).__init__()
        # 定义高频和低频特征提取的卷积层
        self.high_freq_conv = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.high_freq_bn = nn.BatchNorm2d(64)
        self.high_freq_relu = nn.ReLU(inplace=True)

        self.low_freq_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.low_freq_bn = nn.BatchNorm2d(64)
        self.low_freq_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.shape[0]  # 获取批次大小
        high_freq_list, low_freq_list = [], []

        for i in range(batch_size):
            img = x[i].cpu().detach().numpy()  # 处理单张图片
            coeffs2 = pywt.dwt2(img, 'haar')
            LL, (LH, HL, HH) = coeffs2

            # 将小波变换结果转回 PyTorch 张量
            LL = torch.tensor(LL).to(x.device).float()
            LH = torch.tensor(LH).to(x.device).float()
            HL = torch.tensor(HL).to(x.device).float()
            HH = torch.tensor(HH).to(x.device).float()

            # 拼接高频部分
            high_freq = torch.cat([LH, HL, HH], dim=0)
            low_freq = LL

            high_freq_list.append(high_freq.unsqueeze(0))  # 保持批次维度
            low_freq_list.append(low_freq.unsqueeze(0))

        # 堆叠批次
        high_freq = torch.cat(high_freq_list, dim=0)
        low_freq = torch.cat(low_freq_list, dim=0)

        # 处理高频和低频特征
        high_freq = self.high_freq_conv(high_freq)
        high_freq = self.high_freq_bn(high_freq)
        high_freq = self.high_freq_relu(high_freq)

        low_freq = self.low_freq_conv(low_freq)
        low_freq = self.low_freq_bn(low_freq)
        low_freq = self.low_freq_relu(low_freq)

        return high_freq, low_freq



class DualNet(nn.Module):
    def __init__(self, cfg):
        super(DualNet, self).__init__()
        self.forward_count = 0#引入计数器
        self.iter_count = 0
        # 使用现有的ResNet并加载预训练权重
        self.resnet = ResNet(cfg)
        # 定义小波网络
        self.wavelet_net = WaveletNet()
        # 定义独立的 CBAM 注意力模块
        self.cbam_layers_x = nn.ModuleDict()  # 为 x 创建 CBAM 层
        self.cbam_layers_high = nn.ModuleDict()  # 为 wavelet_high_resized 创建 CBAM 层
        self.cbam_layers_low = nn.ModuleDict()  # 为 wavelet_low_resized 创建 CBAM 层
        self.fusion_convs = nn.ModuleDict()
        # for name, param in self.named_parameters():
        #     print(f"Layer: {name}, abcdefg requires_grad: {param.requires_grad}")

    def generate_cam(self, feature_map):
        # 对于输入的特征图生成 CAM
        b, c, h, w = feature_map.shape
        print(f"Batch size (b): {b}, Channels (c): {c}, Height (h): {h}, Width (w): {w}")
        print(f"Before avgpool: feature_map shape = {feature_map.shape}")

        # # 使用全局平均池化进行类激活图生成
        # cam = F.adaptive_avg_pool2d(feature_map, (1, 1))#是不是我这里的全局平均池化出了问题?我不该把宽和高压缩为1
        # cam = cam.view(b, c)

        # 打印调试信息，检查形状
        print(f"Before mean: cam shape = {feature_map.shape}")
        # 生成类激活图，注意这里可以根据实际任务调整权重等
        cam = torch.mean(feature_map, dim=1)
        # 再次打印调试信息，检查形状是否正确
        print(f"After mean: cam shape = {cam.shape}")
        # 自动获取高度和宽度，确保与cam形状匹配
        # b, c, h_new, w_new = feature_map.shape
        # cam = cam.view(b, h_new, w_new)
        # 归一化处理，方便可视化
        cam -= cam.min()
        cam /= cam.max()

        return cam

    def save_cam_image(self, cam, layer_idx, count,iter_count):
        import matplotlib.pyplot as plt
        import os

        # 如果 cam 是三维的，取平均值或某个通道，确保它是二维的
        if cam.ndim == 3:
            cam = torch.mean(cam, dim=0)  # 对通道维度进行平均，转换为二维

        # 创建保存目录
        save_dir = './cam_outputs/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 绘制并保存 CAM 图像
        plt.imshow(cam.detach().cpu().numpy(), cmap='jet')
        plt.colorbar()
        plt.title(f"CAM for FPN Layer {layer_idx} at Iteration {count}")
        plt.savefig(os.path.join(save_dir, f'cam_layer_{layer_idx}_iter_{count}_iter_count_{iter_count}.png'))
        plt.close()

    def forward(self, x):
        # 增加计数器

        if not self.training:
            self.forward_count += 1
        else:
            # 如果是训练模式，清零计数器
            self.forward_count = 0
        self.iter_count += 1

        outputs = []

            # 通过小波网络

        # 通过小波网络获取高频和低频特征
        wavelet_high, wavelet_low = self.wavelet_net(x)

        # 打印输入的初始信息
        # print(f"input ResNetx的形状: {x.shape}")
        # print("input feature x - Mean: {:.4f}, Std: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
        #     x.mean().item(), x.std().item(), x.max().item(), x.min().item()))

        # 打印小波特征的信息
        # print("Wavelet High Frequency Features - Mean: {:.4f}, Std: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
        #     wavelet_high.mean().item(), wavelet_high.std().item(), wavelet_high.max().item(),
        #     wavelet_high.min().item()))
        # print("Wavelet Low Frequency Features - Mean: {:.4f}, Std: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
        #     wavelet_low.mean().item(), wavelet_low.std().item(), wavelet_low.max().item(), wavelet_low.min().item()))
        # print(f"wavelet_high的形状: {wavelet_high.shape}")
        # print(f"wavelet_low的形状: {wavelet_low.shape}")

        # 通过ResNet的stem提取特征
        x = self.resnet.stem(x)
        # print("stem feature x - Mean: {:.4f}, Std: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
        #     x.mean().item(), x.std().item(), x.max().item(), x.min().item()))
        # print(f"x_feature的形状: {x.shape}")


        # 遍历ResNet的各个阶段
        for stage_name in self.resnet.stages:
            # 通过ResNet阶段
            x = getattr(self.resnet, stage_name)(x)
            # 打印当前阶段名称和 x 的形状
            # print(f"Stage: {stage_name}, x shape: {x.shape}")
            # print("Stage x - Mean: {:.4f}, Std: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            #     x.mean().item(), x.std().item(), x.max().item(), x.min().item()))

            # 如果需要返回该阶段的特征
            # if self.resnet.return_features[stage_name]:
            if stage_name == 'layer2':
                # 调整大小，确保wavelet特征与当前ResNet阶段的输出大小匹配
                wavelet_high_resized = F.interpolate(wavelet_high, size=x.shape[2:], mode='bilinear',
                                                     align_corners=False)
                wavelet_low_resized = F.interpolate(wavelet_low, size=x.shape[2:], mode='bilinear', align_corners=False)

                # 打印调整后的小波特征的信息
                # print(
                #     "Wavelet High Frequency Features Resized - Mean: {:.4f}, Std: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                #         wavelet_high_resized.mean().item(), wavelet_high_resized.std().item(),
                #         wavelet_high_resized.max().item(),
                #         wavelet_high_resized.min().item()))
                # print(
                #     "Wavelet Low Frequency Features Resized - Mean: {:.4f}, Std: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                #         wavelet_low_resized.mean().item(), wavelet_low_resized.std().item(),
                #         wavelet_low_resized.max().item(),
                #         wavelet_low_resized.min().item()))

                # 为 x、wavelet_high_resized 和 wavelet_low_resized 分别创建独立的 CBAM 增强
                if stage_name not in self.cbam_layers_x:
                    self.cbam_layers_x[stage_name] = CBAM(x.shape[1], x.shape[1]).to(x.device)

                if stage_name not in self.cbam_layers_high:
                    self.cbam_layers_high[stage_name] = CBAM(wavelet_high_resized.shape[1], x.shape[1]).to(wavelet_high_resized.device)

                if stage_name not in self.cbam_layers_low:
                    self.cbam_layers_low[stage_name] = CBAM(wavelet_low_resized.shape[1], x.shape[1]).to(wavelet_low_resized.device)

                # 分别对 x、wavelet_high_resized、wavelet_low_resized 进行 CBAM 增强
                # x_cbam = self.cbam_layers_x[stage_name](x)
                # print(f"x_cbam的形状: {x_cbam.shape}")

                wavelet_high_cbam = self.cbam_layers_high[stage_name](wavelet_high_resized)
                # print(f"wavelet_high_cbam的形状: {wavelet_high_cbam.shape}")
                # print(
                #     "wavelet_high_cbam的 - Mean: {:.4f}, Std: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                #         wavelet_high_cbam.mean().item(), wavelet_high_cbam.std().item(),
                #         wavelet_high_cbam.max().item(),
                #         wavelet_high_cbam.min().item()))

                wavelet_low_cbam = self.cbam_layers_low[stage_name](wavelet_low_resized)
                # print(f"wavelet_low_cbam的形状: {wavelet_low_cbam.shape}")
                # print(
                #     "wavelet_low_cbam的 - Mean: {:.4f}, Std: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                #         wavelet_low_cbam.mean().item(), wavelet_low_cbam.std().item(),
                #         wavelet_low_cbam.max().item(),
                #         wavelet_low_cbam.min().item()))
                # 进行残差拼接：x + wavelet_high_cbam + wavelet_low_cbam
                # fused_feature = x_cbam + wavelet_high_cbam + wavelet_low_cbam
                fused_feature = x + wavelet_high_cbam + wavelet_low_cbam


                # 打印融合后的特征信息
                # print(f"fused_feature的形状: {fused_feature.shape}")
                # print(
                #     f"fused_feature - Mean: {fused_feature.mean().item():.4f}, Std: {fused_feature.std().item():.4f}, Max: {fused_feature.max().item():.4f}, Min: {fused_feature.min().item():.4f}")

                # 添加 ReLU 激活
                fused_feature = F.relu(fused_feature)
                # print(
                #     f"fused3_feature - Mean: {fused_feature.mean().item():.4f}, Std: {fused_feature.std().item():.4f}, Max: {fused_feature.max().item():.4f}, Min: {fused_feature.min().item():.4f}")

                # 将融合后的特征存储到输出列表中
                outputs.append(fused_feature)
            elif self.resnet.return_features[stage_name]:
                # 在其他层返回特征，但不进行小波融合
                outputs.append(x)
            #
            # if stage_name == 'layer4':
            #     # FPN有4个输出，需要为每个输出计算CAM
            #     for i, fpn_output in enumerate(outputs):
            #         if self.forward_count % 100 == 0:
            #             # 每 1000 次运行时输出CAM
            #             print(f"outputs的形状: {fpn_output.shape}")
            #             cam = self.generate_cam(fpn_output)
            #             print(f"cam的形状: {cam.shape}")
            #             print(f"Generating CAM for FPN layer {i + 1} at iteration {self.forward_count}")
            #             # 这里可以将 CAM 可视化或存储
            #             self.save_cam_image(cam, i + 1, self.forward_count)
                # 在测试模式下输出 CAM
            if not self.training and stage_name == 'layer4':
                # 控制输出 CAM 的间隔
                if self.forward_count == 100:
                    for i, fpn_output in enumerate(outputs):
                        print(f"outputs的形状: {fpn_output.shape}")
                        cam = self.generate_cam(fpn_output)
                        print(f"cam的形状: {cam.shape}")
                        print(f"Generating CAM for FPN layer {i + 1} at iteration {self.forward_count}")
                        self.save_cam_image(cam, i + 1, self.forward_count,self.iter_count)

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
