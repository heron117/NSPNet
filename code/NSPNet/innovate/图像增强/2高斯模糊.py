# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
from stft_core.structures.bounding_box import BoxList
#定义了图像预处理和数据增强的类,用于构建图像变换流水线,类之间可以组合使用,以便在训练深度学习模型时对图像进行一系列的预处理和增强操作


#作者原本是做了一些调整图像大小 随机水平翻转 随机垂直翻转 颜色抖动  转换为张量 标准化的工作
#图像增强 加强在模糊和噪声 和尺度不一的小样本上的表现力。可以增加noise,适当加些训练的噪声。高斯模糊,不能加椒盐噪声 会有黑白点。提升在未见过的数据集上的性能,开发模型的潜在性能。
from PIL import Image
import torch

class Compose(object):  # 定义类将多个变换操作组合在一起，实现数据的逐步增强操作
    def __init__(self, transforms):
        self.transforms = transforms  # 将输入的变换操作列表存储在 self.transforms 中，以供后续使用

    def inspect_target(self, target, transform):
        """检查并打印 target 的数据类型和相关信息."""
        if target is None:
            print(f"After {transform}: target is None")
        elif hasattr(target, "bbox") and hasattr(target, "size"):
            print(f"After {transform}: target is a bounding box-like object with bbox {target.bbox} and size {target.size}")
            print("Bounding box shape:", target.bbox.shape)
            print(target)
            # 检查并打印 labels 字段
            if hasattr(target, "get_field") and target.has_field("labels"):
                labels = target.get_field("labels")
                print("Labels:", labels)
                print("Labels shape:", labels.shape)
            else:
                print("No labels field found in target.")
        else:
            print(f"After {transform}: target is of unknown type {type(target)}")

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)  # 将当前变换操作 t 应用于 image 和 target
            # self.inspect_target(target, t)  # 调用 inspect_target 打印 target 信息
        return image, target  # 返回所有变换操作应用后的图像和标签

    def __repr__(self):  # 定义 __repr__ 方法，用于返回类的字符串表示，便于调试
        format_string = self.__class__.__name__ + "(\n"
        for t in self.transforms:
            format_string += "    {0}\n".format(t)
        format_string += ")"
        return format_string


class Resize(object):#用于调整图像大小,根据输入图像的大小和配置的最小最大尺寸重新计算新的尺寸
    def __init__(self, min_size, max_size):#min_size 和 max_size 分别为图像调整的最小和最大尺寸  MIN_SIZE_TRAIN: (800,)
        if not isinstance(min_size, (list, tuple)):#判断 min_size 是否为列表或元组。
            min_size = (min_size,)#如果 min_size 不是列表或元组，将其转为元组，以支持多种尺寸选择。
        self.min_size = min_size#存储 min_size 和 max_size 为类的属性。
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):#定义 get_size 方法，用于根据图像的原始尺寸计算新的尺寸。
        w, h = image_size#将图像的宽度和高度分别赋值给 w 和 h。
        size = random.choice(self.min_size)#随机选择 self.min_size 中的一个值作为新尺寸的基础。
        max_size = self.max_size#获取 self.max_size。
        if max_size is not None:#如果 max_size 存在，进行进一步检查。
            min_original_size = float(min((w, h)))#获取原图像的最小边长。
            max_original_size = float(max((w, h)))#获取原图像的最大边长。
            if max_original_size / min_original_size * size > max_size:#判断如果新尺寸超过了 max_size，需要对 size 进行调整。
                size = int(round(max_size * min_original_size / max_original_size))#如果超出 max_size，根据比例计算调整后的尺寸。

        if (w <= h and w == size) or (h <= w and h == size):#判断如果图像一边的尺寸已达到目标 size，则返回当前尺寸。
            return (h, w)

        if w < h:
            ow = size#如果图像宽小于高，设置宽为 size，按比例计算高度 oh。
            oh = int(size * h / w)
        else:
            oh = size#如果宽大于高，设置高为 size，按比例计算宽度 ow。
            ow = int(size * w / h)

        #for stft:
        #input image can't smaller than dcn kernel
        #so image scale >= 128*3
        ow = int(max(384, ow))#为了适应某些要求，将宽高调整到至少 384 的尺寸
        oh = int(max(384, oh))

        return (oh, ow)


#__call__使得 Resize 类的实例可以像函数一样调用。
    def __call__(self, image, target=None):#应用调整后的尺寸并返回调整后的图像和目标--这里可能是非常的有趣的
        size = self.get_size(image.size)#获取新的图像尺寸。
        image = F.resize(image, size)#使用 torchvision.transforms.functional.resize 对图像进行调整。
        if target is None:
            return image, target#如果没有 target，直接返回调整后的图像。
        target = target.resize(image.size)#如果存在 target，对目标框也同步进行 resize 操作。
        return image, target#返回调整后的图像和目标框。
    #如果图像增强改变了图像大小,目标框的尺寸和位置会需要同步改变,以保证目标框与增强后的图像保持一致 也使用了resize方法






import random
import numpy as np
import albumentations as A
import cv2
from PIL import Image
import torch
from stft_core.structures.bounding_box import BoxList

class RandomScaleAlbumentations:
    def __init__(self, scale=0.2, to_tensor=True, prob=0.5):
        self.scale = scale
        self.to_tensor = to_tensor
        self.prob = prob  # 应用概率

    def __call__(self, image, targets=None):
        if random.random() < self.prob:  # 根据概率决定是否应用缩放
            return image, targets

        # 如果输入是 PIL 图像，转换为 NumPy 格式
        if isinstance(image, Image.Image):
            image = np.array(image)

        output_height, output_width = image.shape[:2]

        # 根据 targets 是否为 None 决定是否使用 bbox_params
        if targets is None:
            transform = A.Compose([
                A.RandomScale(scale_limit=self.scale, p=1),
                A.PadIfNeeded(min_height=output_height, min_width=output_width, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.CenterCrop(height=output_height, width=output_width),
            ])
            transformed = transform(image=image)
            transformed_image = transformed['image']

            # 如果需要返回 PIL.Image 格式
            if isinstance(transformed_image, np.ndarray):
                transformed_image = Image.fromarray(transformed_image)

            return transformed_image, None

        else:
            # 设置 bbox_params 和 transform
            transform = A.Compose([
                A.RandomScale(scale_limit=self.scale, p=1),
                A.PadIfNeeded(min_height=output_height, min_width=output_width, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.CenterCrop(height=output_height, width=output_width),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1))

            if isinstance(targets, BoxList):
                bboxes = targets.bbox.tolist()
                # 提取标签
                labels = targets.get_field("labels") if targets.has_field("labels") else None
            else:
                raise TypeError("Expected target to be a BoxList instance")

            # 应用增强变换
            transformed = transform(image=image, bboxes=bboxes, labels=labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_labels = transformed['labels']
            print("Transformed bboxes before clipping:", transformed_bboxes)

            # 裁剪边界框以保持在图像范围内
            clipped_bboxes = []
            for bbox in transformed_bboxes:
                xmin, ymin, xmax, ymax = bbox
                xmin = max(0, min(xmin, output_width))
                ymin = max(0, min(ymin, output_height))
                xmax = max(0, min(xmax, output_width))
                ymax = max(0, min(ymax, output_height))
                if xmax > xmin and ymax > ymin:
                    clipped_bboxes.append([xmin, ymin, xmax, ymax])

            # 如果所有边界框都消失，返回空 target
            if not clipped_bboxes:
                transformed_target = BoxList(torch.empty((0, 4)), (output_width, output_height), mode="xyxy")
            else:
                transformed_target = BoxList(torch.tensor(clipped_bboxes), (output_width, output_height), mode="xyxy")
                # 重新添加标签字段
                if labels is not None:
                    transformed_target.add_field("labels", torch.tensor(transformed_labels))

            # 如果需要返回 PIL.Image 格式
            if isinstance(transformed_image, np.ndarray):
                transformed_image = Image.fromarray(transformed_image)

            return transformed_image, transformed_target


import random
from PIL import Image, ImageFilter

class GaussianBlur(object):  # 用于对图像加入高斯模糊
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), prob=0.5):
        self.kernel_size = kernel_size  # 高斯核大小，未使用，但保留作兼容性
        self.sigma = sigma  # 高斯模糊的标准差范围
        self.prob = prob  # 高斯模糊的应用概率

    def __call__(self, image, target=None):
        if random.random() < self.prob:  # 以50%概率应用高斯模糊
            sigma = random.uniform(self.sigma[0], self.sigma[1])  # 随机选择一个sigma值

            # 仅使用 PIL 的 ImageFilter 进行高斯模糊
            if isinstance(image, Image.Image):  # 如果是 PIL 图像
                image = image.filter(ImageFilter.GaussianBlur(sigma))
            else:
                raise TypeError("Expected input type to be PIL.Image")

        return image, target


class RandomHorizontalFlip(object):#用于随机水平翻转图像
    def __init__(self, prob=0.5):#prob=0.5翻转的概念为0.5
        self.prob = prob
        self.chance = 0.0

    def __call__(self, image, target=None):#__call__：对图像进行随机水平翻转。生成一个0到1之间的随机数，如果小于prob，则对图像进行水平翻转
        if target is not None:
            self.chance = random.random()
        if self.chance < self.prob:
            image = F.hflip(image)#F.hflip(image)：使用torchvision.transforms.functional模块中的hflip方法，将图像水平翻转。
            if target is not None:
                target = target.transpose(0)#target.transpose(0)：如果有目标标签，则水平翻转标签。transpose(0)假设目标是一个边界框，翻转对应于坐标系的水平变换。

        return image, target


class RandomVerticalFlip(object):#用于随机垂直翻转图像
    def __init__(self, prob=0.5):#prob：垂直翻转的概率，默认为0.5。

        self.prob = prob

    def __call__(self, image, target=None):#__call__：对图像进行随机垂直翻转。生成一个0到1之间的随机数，如果小于prob，则对图像进行垂直翻转。
        if random.random() < self.prob:
            image = F.vflip(image)#F.vflip(image)：使用vflip方法将图像垂直翻转
            if target is not None:
                target = target.transpose(1)#target.transpose(1)：如果有目标标签，则垂直翻转标签。transpose(1)同样适用于垂直翻转中的坐标变换。
        return image, target


class ColorJitter(object):#随机改变图像的亮度 对比度 饱和度 色调
    def __init__(self,
                 brightness=None,#brightness、contrast、saturation、hue：这些参数控制亮度、对比度、饱和度和色调的变化幅度。
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(#颜色抖动的核心操作，随机改变图像的亮度、对比度、饱和度和色调，从而产生轻微的色彩变动，帮助增强模型的色彩适应性。
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target=None):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):#将图像转换为张量  #将PIL图像或numpy.ndarray格式的图像转换为torch.Tensor格式，使得数据可以直接输入到模型中
    def __call__(self, image, target=None):
        return F.to_tensor(image), target#使用F.to_tensor方法，将图像值从0-255范围缩放到0-1之间，并转换为张量形式。


class Normalize(object):#用于标准化图像
    def __init__(self, mean, std, to_bgr255=True):
        #mean：每个通道的均值，用于标准化。std：每个通道的标准差，用于标准化。to_bgr255：是否将图像从RGB转换为BGR格式，并将像素值放大到0-255范围。
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)#将图像的每个通道的像素值减去均值后除以标准差，将像素值转化为零均值、单位方差的形式，这有助于加速模型的训练收敛。
        if target is None:
            return image, target
        return image, target
#这段代码定义了一些常用的图像变换类，这些类可以组合在一起以便在训练深度学习模型时对图像进行预处理和增强。通过组合这些类，可以构建一个复杂的图像预处理流水线，以提高模型的泛化能力和性能。