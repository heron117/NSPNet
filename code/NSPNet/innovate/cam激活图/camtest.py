# -*- coding: utf-8 -*-
'''
使用CAM 将测试结果可视化——————————单张图像
'''
import numpy as np#导入必要的库
from keras.models import Model
from keras import activations
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops
from PIL import Image
import cv2
import utils
from keras.models import Sequential
from keras.layers.core import Lambda
#使用类激活图技术,将模型的预测结果在单张图像上可视化,
#模型加载 图像处理 计算激活图 将其叠加到输入图像上生成可视化结果

#定义模型训练和图像处理的默认配置参数 模型名称 训练 和验证数据路径 批量大小 学习率 图像形状 和类别数量
class DefaultConfig():
    model_name = 'fsnet_05'
    train_data_path = '.\\dataset\\train\\'
    val_data_path = '.\\dataset\\test\\'
    checkpoints = '.\\checkpoints\\'
    modelpath = '\\model\\'
    fine_tune_model = '\\model\\'
    normal_size = 64
    epochs = 100
    batch_size = 32
    channles = 3  # or 3 or 1
    lr = 0.001
    lr_reduce_patience = 20
    early_stop_patience = 50  # 提前终止训练的步长
    finetune = False
    monitor = 'val_loss'
    image_shape = (128, 128, 3)
    classNumber = 6  # see dataset/tri


if __name__ == '__main__':
#有一个自定义的模型文件
    from keras.models import load_model
    from my_models import baseline, baseline_seblock
#此处假设有一个自定义的模型文件 my_models.py，其中定义了模型结构。之后，代码会加载模型和预训练的权重。
    img_shape = (128, 128, 3)
    mask_shape = (8, 8, 1)
    num_class = 6

    # 载入模型和权重
    config = DefaultConfig()
    baseline_model = baseline(config, [1, 2, 2, 2], scope='baseline')
    baseline_weight = 'E:/datasets/Rcam-plusMelangerTaile/8KLSBackWindow/trained_model/baseline/modelpath/baseline.h5'
    model = load_model(baseline_weight)
    print('------Baseline load weight done------')
#baseline 是自定义模型，并加载预训练权重文件 baseline.h5。成功加载模型后，输出提示“模型加载完成”。
    # 载入图像——图像处理
    img_path = 'E:/datasets/Rcam-plusMelangerTaile/8KLSBackWindow/picked_out/1/755_2019_9_27.bmp'
    img_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img_bgr.shape[:2]
    if h != 128 or w != 128:
        img_bgr = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_LINEAR)
    resnetv1_inputs = np.array([img_bgr])
    # v1 = resnetv1_inputs.squeeze()

    pred = model.predict(resnetv1_inputs)
#预测类别
    class_idx = np.argmax(pred[0])

#计算cam激活图
    class_output = model.output[:, class_idx]

    last_conv_layer = model.get_layer("conv2d_18")  # 网络最后一个卷积层的名字
    gap_weights = model.get_layer("global_average_pooling2d_1")  # 网络GAP操作层的名字
#class_output 获取了与预测类别相关的模型输出。
# last_conv_layer 获取模型的最后一个卷积层（这里假设名称为 conv2d_18）。
# gap_weights 获取模型的全局平均池化层（GAP层）的输出，用于后续计算。
#
    grads = K.gradients(class_output, gap_weights.output)[0]
    iterate = K.function([model.input], [grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([resnetv1_inputs])
    pooled_grads_value = np.squeeze(pooled_grads_value, axis=0)
#grads 计算了类别输出对于GAP层输出的梯度，表明类别对特征的依赖程度。
# iterate 是一个Keras后端函数，用于计算梯度和卷积层输出。
# pooled_grads_value 获取梯度值并通过全局池化缩小维度。
# conv_layer_output_value 获取卷积层的输出。
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#此处遍历卷积层的通道，调整每个通道的输出值，并生成平均热力图。通过ReLU激活函数和归一化操作，保证热力图的值处于合理范围内。
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # relu激活。
    heatmap /= np.max(heatmap)
    #
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
    # img = img_to_array(image)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)  # 将cam图像叠加到原图上
    cv2.imwrite('image-cam.png', superimposed_img)
#最后，热力图调整为原图的尺寸，并通过 cv2.applyColorMap 上色。将生成的CAM图与原图叠加，保存为 image-cam.png，完成可视化。