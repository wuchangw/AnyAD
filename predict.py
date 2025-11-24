import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger, \
    evaluation_batch_with_feature_saving, evaluation_batch_with_visualization, cal_anomaly_maps, get_gaussian_kernel
from torchvision import transforms
from torch.utils.data import DataLoader
from models import vit_encoder
from models.model import INP_Former, StableAdamW, Mlp, Aggregation_Block, Prototype_Block
from torch.utils.data import Dataset
import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import torch.multiprocessing
import json

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()
    ])
    return data_transforms, gt_transforms


class ModalityMasker:
    """模态掩码工具类，用于处理不同的模态组合"""
    MODALITY_COMBINATIONS = {
        1: [1, 0, 0],  # 仅flair
        2: [0, 1, 0],  # 仅t1
        3: [0, 0, 1],  # 仅t2
        4: [1, 1, 0],  # flair + t1
        5: [1, 0, 1],  # flair + t2
        6: [0, 1, 1],  # t1 + t2
        7: [1, 1, 1]  # 全模态
    }

    def __init__(self, combination_id=7):
        self.combination = self.MODALITY_COMBINATIONS.get(combination_id, [1, 1, 1])

    def mask_modalities(self, img_3ch):
        """
        根据设定的组合掩码模态
        img_3ch: (B, C, H, W) 三通道图像，通道顺序为[flair, t1, t2]
        返回: 掩码后的图像
        """
        masked_img = img_3ch.clone()
        for i, mask in enumerate(self.combination):
            if mask == 0:  # 掩码该模态
                masked_img[:, i] = 0.0  # 将该模态置为0
        return masked_img


class SingleImagePredictor:
    """单张图片预测器"""

    def __init__(self, model, transform, device, combination_id=7, max_ratio=0.01, resize_mask=256):
        self.model = model
        self.transform = transform
        self.device = device
        self.masker = ModalityMasker(combination_id)
        self.combination_id = combination_id
        self.max_ratio = max_ratio
        self.resize_mask = resize_mask
        self.gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    def get_required_modality_count(self):
        """根据组合ID返回需要的模态数量"""
        if self.combination_id <= 3:
            return 1
        elif self.combination_id <= 6:
            return 2
        else:  # combination_id == 7
            return 3

    def get_modality_names(self):
        """根据组合ID返回需要的模态名称"""
        if self.combination_id == 1:
            return ["flair"]
        elif self.combination_id == 2:
            return ["t1"]
        elif self.combination_id == 3:
            return ["t2"]
        elif self.combination_id == 4:
            return ["flair", "t1"]
        elif self.combination_id == 5:
            return ["flair", "t2"]
        elif self.combination_id == 6:
            return ["t1", "t2"]
        else:  # combination_id == 7
            return ["flair", "t1", "t2"]

    def load_single_image(self, image_paths):
        """
        加载单张图片的多模态数据
        image_paths: 图片路径列表，根据组合ID确定数量
        """
        required_count = self.get_required_modality_count()
        modality_names = self.get_modality_names()

        if len(image_paths) != required_count:
            raise ValueError(f"组合ID {self.combination_id} 需要 {required_count} 个模态图片路径: {modality_names}")

        # 创建完整的三个模态的数组（会通过掩码处理）
        modality_images = []
        available_modalities = ['flair', 't1', 't2']

        for modality in available_modalities:
            if modality in modality_names:
                # 找到对应的图片路径
                idx = modality_names.index(modality)
                img_path = image_paths[idx]
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"图片不存在: {img_path}")

                img = Image.open(img_path).convert('L')
                img_np = np.array(img).astype(np.float32)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                modality_images.append(img_np)
            else:
                # 对于不存在的模态，创建空白图像
                if modality_images:  # 如果已经有图像，使用第一个图像的尺寸
                    blank_img = np.zeros_like(modality_images[0])
                else:
                    # 如果没有图像，需要先加载一个来获取尺寸
                    img_path = image_paths[0]
                    img = Image.open(img_path).convert('L')
                    img_np = np.array(img).astype(np.float32)
                    blank_img = np.zeros_like(img_np)
                modality_images.append(blank_img)

        img_3ch = np.stack(modality_images, axis=-1)
        img_pil = Image.fromarray((img_3ch * 255).astype(np.uint8))
        img_tensor = self.transform(img_pil)

        # 应用模态掩码
        img_tensor = self.masker.mask_modalities(img_tensor.unsqueeze(0))  # 添加batch维度

        return img_tensor.to(self.device), image_paths[0]  # 返回第一个路径作为标识

    def predict(self, image_paths):
        """
        对单张图片进行预测
        返回: 预测结果字典
        """
        self.model.eval()

        with torch.no_grad():
            # 加载图片
            img_tensor, img_path = self.load_single_image(image_paths)

            # 模型预测 - 使用与evaluation_batch相同的逻辑
            output = self.model(img_tensor)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img_tensor.shape[-1])

            # 调整大小
            if self.resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=self.resize_mask, mode='bilinear', align_corners=False)

            # 高斯平滑
            anomaly_map = self.gaussian_kernel(anomaly_map)

            # 计算图像级异常分数
            if self.max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map_flat = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:,
                           :int(anomaly_map_flat.shape[1] * self.max_ratio)]
                sp_score = sp_score.mean(dim=1)

            # 解析输出
            result = {
                'image_path': img_path,
                'image_tensor': img_tensor,
                'anomaly_score': sp_score.item(),  # 图像级异常分数
                'anomaly_map': anomaly_map.squeeze().cpu().numpy(),  # 像素级异常热力图
                'combination_id': self.combination_id,
                'modality_names': self.get_modality_names(),
                'outputs': output
            }

            return result


def load_model(args, device):
    """加载训练好的模型"""
    # 模型初始化
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]

    # 模型组件
    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim))])

    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])

    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])

    model = INP_Former(
        encoder=encoder, bottleneck=Bottleneck,
        aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
        target_layers=target_layers, remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=INP
    ).to(device)

    # 加载训练好的权重
    model_path = os.path.join(args.save_dir, args.save_name, 'queshi_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型权重已加载: {model_path}")
    else:
        print(f"警告: 未找到模型权重文件 {model_path}")

    return model


import matplotlib.pyplot as plt
import cv2

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_heatmap(result, save_dir, save_name, threshold):
    """保存热力图和叠加图"""
    try:
        # 创建保存目录
        heatmap_dir = os.path.join(save_dir, save_name, 'heatmaps_MU_INPFormer')
        os.makedirs(heatmap_dir, exist_ok=True)

        # 获取基本信息
        heatmap = result['anomaly_map']
        original_filename = os.path.basename(result['image_path'])
        filename_no_ext = os.path.splitext(original_filename)[0]
        combination_id = result['combination_id']

        # 获取原图（使用第一个模态的图像）
        img_path = result['image_path']
        original_img = Image.open(img_path).convert('L')
        original_img = original_img.resize((heatmap.shape[1], heatmap.shape[0]))
        original_img_np = np.array(original_img)

        # 创建脑部区域的mask（基于原图的非零像素）
        brain_mask = original_img_np > 0

        binary_map = (heatmap > threshold).astype(np.float32)
        binary_map[~brain_mask] = 0  # 非脑部区域设为0

        # 应用脑部mask到热力图，非脑部区域设为0
        heatmap_masked = heatmap.copy()
        heatmap_masked[~brain_mask] = 0

        # 保存热力图（只显示脑部区域）
        plt.figure(figsize=(2.56, 2.56), facecolor='black')  # 设置背景为黑色
        plt.imshow(heatmap_masked, cmap='jet')
        plt.axis('off')
        heatmap_path = os.path.join(heatmap_dir, f'MU_{filename_no_ext}_combo{combination_id}_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 保存叠加图（只显示脑部区域的热力图叠加）
        plt.figure(figsize=(2.56, 2.56))
        plt.imshow(original_img_np, cmap='gray')
        # 只叠加脑部区域的热力图
        plt.imshow(heatmap_masked, cmap='jet', alpha=0.5)
        plt.axis('off')
        overlay_path = os.path.join(heatmap_dir, f'MU_{filename_no_ext}_combo{combination_id}_overlay.png')
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 保存二值阈值图（只显示脑部区域）
        plt.figure(figsize=(2.56, 2.56))
        plt.imshow(binary_map, cmap='gray')
        plt.axis('off')
        binary_path = os.path.join(heatmap_dir, f'MU_{filename_no_ext}_combo{combination_id}_binary.png')
        plt.savefig(binary_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        print_fn(f"热力图已保存: {heatmap_path}")
        print_fn(f"叠加图已保存: {overlay_path}")
        print_fn(f"二值图已保存: {binary_path}")

    except Exception as e:
        print_fn(f"保存热力图时出错: {e}")
        import traceback
        traceback.print_exc()

def main(args):
    setup_seed(3)

    # 加载数据预处理
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    # 加载模型
    model = load_model(args, device)

    # 创建预测器
    predictor = SingleImagePredictor(
        model=model,
        transform=data_transform,
        device=device,
        combination_id=args.combination_id,
        max_ratio=0.01,
        resize_mask=256
    )

    required_count = predictor.get_required_modality_count()
    modality_names = predictor.get_modality_names()

    print_fn(f'使用模态组合 {args.combination_id}')
    print_fn(f'需要 {required_count} 个模态: {modality_names}')
    print_fn(f'基础数据路径: {args.base_data_path}')

    # 单张图片预测
    if args.single_image_paths:
        # 处理相对路径，组合成完整路径
        relative_paths = [path.strip() for path in args.single_image_paths.split(',')]
        full_paths = []

        for i, rel_path in enumerate(relative_paths):
            if rel_path.startswith('/'):
                # 如果是绝对路径，直接使用
                full_paths.append(rel_path)
            else:
                # 如果是相对路径，基于基础路径组合
                modality = modality_names[i] if i < len(modality_names) else 'flair'
                full_path = os.path.join(args.base_data_path, modality, rel_path)
                full_paths.append(full_path)

        print_fn(f"完整图片路径: {full_paths}")

        try:
            result = predictor.predict(full_paths)
            print_fn(f"预测完成!")
            print_fn(f"图片: {os.path.basename(result['image_path'])}")
            print((result['image_tensor']).shape)
            print_fn(f"模态组合: {result['combination_id']}")
            print_fn(f"使用模态: {result['modality_names']}")
            print_fn(f"异常分数: {result['anomaly_score']:.4f}")
            print_fn(f"热力图形状: {result['anomaly_map'].shape}")

            threshold = 0.34
            if result['anomaly_score'] > threshold:
                print_fn("判断: 异常")
            else:
                print_fn("判断: 正常")

            # 显示热力图统计信息
            heatmap = result['anomaly_map']
            print_fn(
                f"热力图统计 - 最小值: {heatmap.min():.4f}, 最大值: {heatmap.max():.4f}, 均值: {heatmap.mean():.4f}")

            # 保存热力图
            save_heatmap(result, args.save_dir, args.save_name, threshold)

        except Exception as e:
            print_fn(f"预测失败: {e}")
            import traceback
            traceback.print_exc()

    else:
        print_fn("请通过 --single_image_paths 参数提供图片路径")
        print_fn(f"需要提供 {required_count} 个图片路径，对应模态: {modality_names}")
        print_fn("示例:")
        if required_count == 1:
            print_fn("  --single_image_paths 'Brats18_CBICA_AQU_1_slice110.png'")
        elif required_count == 2:
            print_fn("  --single_image_paths 'Brats18_CBICA_AQU_1_slice110.png,Brats18_CBICA_AQU_1_slice110.png'")
        else:
            print_fn(
                "  --single_image_paths 'Brats18_CBICA_AQU_1_slice110.png,Brats18_CBICA_AQU_1_slice110.png,Brats18_CBICA_AQU_1_slice110.png'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据集参数
    parser.add_argument('--modalities', nargs='+', default=['flair', 't1', 't2'],
                        help='List of modalities to use (e.g., --modalities flair t1 t2)')
    parser.add_argument('--combination_id', type=int, default=1,
                        help='Modality combination ID (1-7): 1=flair, 2=t1, 3=t2, 4=flair+t1, 5=flair+t2, 6=t1+t2, 7=all')

    # 在 argparse 中添加基础路径参数
    parser.add_argument('--base_data_path', type=str, default='/home/wuchangwei/wcw/new_data/MU/test/ABNORMAL',
                        help='基础数据路径，图片将基于此路径查找')

    # 单张图片预测参数
    parser.add_argument('--single_image_paths', type=str, default='PatientID_0054_Timepoint_1_slice100.png',
                        help='图片相对路径，用逗号分隔。基于base_data_path查找')

    # 模型参数
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14',
                        help='Encoder architecture')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # 训练参数
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dist_lambda', type=float, default=0.1, help='Weight for the distribution alignment loss')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='/home/wuchangwei/wcw/dinov2_model/result/INPFormer')
    parser.add_argument('--save_name', type=str, default='INPFormer_results_in2018/modality1')

    args = parser.parse_args()

    # 初始化日志
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    # 运行主函数
    results = main(args)