import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger, evaluation_batch_with_feature_saving, evaluation_batch_with_visualization
from torchvision import transforms
from torch.utils.data import DataLoader
from models import vit_encoder
from models.model import INP_Former,StableAdamW,Mlp, Aggregation_Block, Prototype_Block
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
        7: [1, 1, 1]   # 全模态
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


class MedicalDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase, modalities=['flair', 't1', 't2'], combination_id=7,
                 dynamic_combinations=False):
        self.modalities = modalities
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        self.dynamic_combinations = dynamic_combinations

        # 初始化时使用默认组合，但实际组合会在getitem中动态确定
        self.masker = ModalityMasker(combination_id)
        self.img_paths, self.gt_paths, self.labels = self.load_dataset(root)

        # 存储所有可能的组合ID
        self.all_combination_ids = list(range(1, 8))  # 1到7

    def set_combination_id(self, combination_id):
        """动态设置模态组合"""
        self.masker = ModalityMasker(combination_id)

    def load_dataset(self, root):
        img_paths = []
        gt_paths = []
        labels = []

        if self.phase == 'train':
            first_modality_path = os.path.join(root, 'train', self.modalities[0], '*.png')
            sample_ids = [os.path.basename(f).split('.')[0] for f in glob.glob(first_modality_path)]

            for sample_id in sample_ids:
                modality_files = []
                modalities_available = True

                for modality in self.modalities:
                    modality_path = os.path.join(root, 'train', modality, f"{sample_id}.png")
                    if not os.path.exists(modality_path):
                        modalities_available = False
                        break
                    modality_files.append(modality_path)

                if modalities_available:
                    img_paths.append(modality_files)
                    gt_paths.append(None)
                    labels.append(0)

            print(f"Loaded {len(img_paths)} training samples with {len(self.modalities)} modalities each")

        else:
            first_modality_path = os.path.join(root, 'test', 'NORMAL', self.modalities[0], '*.png')
            normal_sample_ids = [os.path.basename(f).split('.')[0] for f in glob.glob(first_modality_path)]

            for sample_id in normal_sample_ids:
                modality_files = []
                modalities_available = True

                for modality in self.modalities:
                    modality_path = os.path.join(root, 'test', 'NORMAL', modality, f"{sample_id}.png")
                    if not os.path.exists(modality_path):
                        modalities_available = False
                        break
                    modality_files.append(modality_path)

                if modalities_available:
                    img_paths.append(modality_files)
                    gt_paths.append(None)
                    labels.append(0)

            first_modality_path = os.path.join(root, 'test', 'ABNORMAL', self.modalities[0], '*.png')
            abnormal_sample_ids = [os.path.basename(f).split('.')[0] for f in glob.glob(first_modality_path)]

            for sample_id in abnormal_sample_ids:
                modality_files = []
                modalities_available = True

                for modality in self.modalities:
                    modality_path = os.path.join(root, 'test', 'ABNORMAL', modality, f"{sample_id}.png")
                    if not os.path.exists(modality_path):
                        modalities_available = False
                        break
                    modality_files.append(modality_path)

                if modalities_available:
                    mask_path = os.path.join(root, 'test', 'ABNORMAL', 'mask', f"{sample_id}_mask.png")
                    if not os.path.exists(mask_path):
                        print(f"Warning: Mask not found for {sample_id}")
                        mask_path = None

                    img_paths.append(modality_files)
                    gt_paths.append(mask_path)
                    labels.append(1)

            print(f"Loaded {len(normal_sample_ids)} normal and {len(abnormal_sample_ids)} abnormal test samples")

        return img_paths, gt_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        modality_images = []
        for modality_path in self.img_paths[idx]:
            img = Image.open(modality_path).convert('L')
            img_np = np.array(img).astype(np.float32)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            modality_images.append(img_np)

        img_3ch = np.stack(modality_images, axis=-1)
        img_pil = Image.fromarray((img_3ch * 255).astype(np.uint8))
        img_tensor = self.transform(img_pil)

        # 应用模态掩码
        img_tensor = self.masker.mask_modalities(img_tensor)

        label = self.labels[idx]
        if label == 1 and self.gt_paths[idx] is not None:
            gt = Image.open(self.gt_paths[idx]).convert('L')
            gt = self.gt_transform(gt)
        else:
            gt = torch.zeros((1, *img_tensor.shape[1:]))

        return img_tensor, gt, label, self.img_paths[idx][0]

def main(args):
    setup_seed(3)
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    test_data = MedicalDataset(
        root=args.data_path,
        transform=data_transform,
        gt_transform=gt_transform,
        phase='test',
        modalities=['t1ce', 't1ce', 't1ce'],
        combination_id=7
    )

    test_dataloader = DataLoader(test_data, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)

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


    if args.phase == 'test':
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'best_model.pth')))
        current_combination = args.combination_id
        print_fn(f'Using modality combination {current_combination}')
        model.eval()
        # results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)

        results = evaluation_batch_with_feature_saving(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)

        # results = evaluation_batch_with_visualization(
        #     model, test_dataloader, device,
        #     max_ratio=0.1,
        #     resize_mask=392,
        #     save_dir=os.path.join(args.save_dir, 'visualizations')
        # )
    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
    print_fn(
        'Results:\n'
        'Image-Level - AUROC: {:.4f}, AP: {:.4f}, F1: {:.4f}\n'
        'Pixel-Level - AUROC: {:.4f}, AP: {:.4f}, F1: {:.4f}, AUPRO: {:.4f}'.format(
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px
        )
    )
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据集参数
    parser.add_argument('--data_path', type=str, default='/home/wuchangwei/wcw/new_data/BraTs2018')
    parser.add_argument('--modalities', nargs='+', default=['flair', 't1', 't2'],
                        help='List of modalities to use (e.g., --modalities flair t1 t2)')
    parser.add_argument('--combination_id', type=int, default=7, help='Modality combination ID (1-7)')
    args = parser.parse_args()
    # 模型参数
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14',
                        help='Encoder architecture')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--total_epochs', type=int, default=300)
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dist_lambda', type=float, default=0.1, help='Weight for the distribution alignment loss')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./result_in_Brats2018_dist0.2')
    parser.add_argument('--save_name', type=str, default='Mult_class_INPFormer')

    args = parser.parse_args()

    # 初始化日志
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    # 运行主函数
    results = main(args)