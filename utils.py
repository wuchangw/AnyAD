import torch
from numpy.random import normal
import random
import logging
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import os
from functools import partial
import math
from tqdm import tqdm
from aug_funcs import rot_img, translation_img, hflip_img, grey_img, rot90_img
import torch.backends.cudnn as cudnn
from adeval import EvalAccumulatorCuda


def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp,
                   use_metrics=['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)

    score_min = min(pr_sp)
    score_max = max(pr_sp)
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()

    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
    accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                           torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))

    # for i in range(torch.tensor(pr_px).size(0)):
    #     accum.add_image(torch.tensor(pr_sp[i]), torch.tensor(gt_sp[i]))

    metrics = accum.summary()
    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            metric_results[metric] = auroc_sp
        elif metric.startswith('I-AP'):
            ap_sp = average_precision_score(gt_sp, pr_sp)
            metric_results[metric] = ap_sp
        elif metric.startswith('I-F1_max'):
            best_f1_score_sp = f1_score_max(gt_sp, pr_sp)
            metric_results[metric] = best_f1_score_sp
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            metric_results[metric] = metrics['p_aupr']
        elif metric.startswith('P-F1_max'):
            best_f1_score_px = f1_score_max(gt_px.ravel(), pr_px.ravel())
            metric_results[metric] = best_f1_score_px
        elif metric.startswith('AUPRO'):
            metric_results[metric] = metrics['p_aupro']
    return list(metric_results.values())


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augmentation(img):
    img = img.unsqueeze(0)
    augment_img = img
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(img, angle)
        augment_img = torch.cat([augment_img, rotate_img], dim=0)
        # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(img, a, b)
        augment_img = torch.cat([augment_img, trans_img], dim=0)
        # hflip img
    flipped_img = hflip_img(img)
    augment_img = torch.cat([augment_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(img)
    augment_img = torch.cat([augment_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(img, angle)
        augment_img = torch.cat([augment_img, rotate90_img], dim=0)
    augment_img = (augment_img[torch.randperm(augment_img.size(0))])
    return augment_img


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x


def global_cosine_hm_adaptive(a, b, y=3):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
        mean_dist = point_dist.mean()
        factor = (point_dist / mean_dist) ** (y)
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)

        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, save_scores_path="/home/wuchangwei/wcw/csv/MU_in_Pretreat_.csv"):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    # 新增：保存每个样本的详细信息
    all_sample_details = []

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)



        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        # 统计信息
        normal_scores = pr_list_sp[gt_list_sp == 0]
        anomaly_scores = pr_list_sp[gt_list_sp == 1]

        #
        # # 保存到CSV文件
        # if save_scores_path:
        #     import pandas as pd
        #     # 保存图像级分数
        #     image_df = pd.DataFrame({
        #         'image_score': pr_list_sp,
        #         'image_label': gt_list_sp
        #     })
        #     image_df.to_csv(save_scores_path.replace('.csv', '1.csv'), index=False)
        #     print(f"图像级分数已保存到: {save_scores_path.replace('.csv', '1.csv')}")
        #

        # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px,
                                                                                  gt_list_sp)


    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch_with_feature_saving(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None,
                                         save_features_dir="/home/wuchangwei/wcw/features",
                                         dataset_name="2018_flair"):
    """带有特征保存的评估函数"""

    # 原有的评估逻辑
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    # 新增：特征收集
    all_features = []
    all_labels = []
    all_anomaly_scores = []
    all_img_paths = []

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]

            # 提取特征
            if isinstance(de, (list, tuple)):
                decoder_features = de[-1]
            else:
                decoder_features = de

            # 全局平均池化
            if len(decoder_features.shape) == 4:
                features_pooled = F.adaptive_avg_pool2d(decoder_features, (1, 1))
                features_pooled = features_pooled.view(features_pooled.size(0), -1)
            elif len(decoder_features.shape) == 3:
                features_pooled = decoder_features.mean(dim=1)
            else:
                features_pooled = decoder_features

            all_features.append(features_pooled.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_img_paths.extend(img_path)

            # 原有的异常检测流程
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
            all_anomaly_scores.append(sp_score.cpu().numpy())

    # 保存特征到CSV
    os.makedirs(save_features_dir, exist_ok=True)
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_anomaly_scores = np.concatenate(all_anomaly_scores, axis=0)

    feature_columns = [f'feature_{i}' for i in range(all_features.shape[1])]
    df = pd.DataFrame(all_features, columns=feature_columns)
    df['label'] = all_labels
    df['anomaly_score'] = all_anomaly_scores
    df['image_path'] = all_img_paths
    df['dataset'] = dataset_name

    feature_csv_path = os.path.join(save_features_dir, f'{dataset_name}_features.csv')
    df.to_csv(feature_csv_path, index=False)
    print(f"特征已保存到: {feature_csv_path}")

    # 原有的评估结果计算
    gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
    pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
    gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
    pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(
        pr_list_px, pr_list_sp, gt_list_px, gt_list_sp
    )

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px], feature_csv_path
def denormalize_grayscale(img):
    """
    专门处理灰度图像的反归一化
    """
    # 假设输入是单通道灰度图像
    std = 0.229
    mean = 0.485

    # 反归一化并转换到0-255范围
    x = (img * std + mean) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)

    return x


def save_comparison_visualization(anomaly_map, gt_mask, original_img, save_path):
    """
    保存异常图、真实掩码和原图的对比可视化
    """
    fig = plt.figure(figsize=(18, 4))
    gs = plt.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.1])

    axes = [plt.subplot(gs[i]) for i in range(4)]
    cax = plt.subplot(gs[4])

    # 原始图像处理
    original_img_np = original_img.cpu().squeeze(0).numpy()

    # 如果已经是3通道复制而成的，直接取第一个通道即可
    if original_img_np.shape[0] == 3:
        # 使用第一个通道（假设3个通道内容相同）
        original_img_gray = original_img_np[0]  # 形状: (H, W)
    else:
        original_img_gray = original_img_np.squeeze()  # 去掉通道维度

    # 确保是2D数组 (H, W)
    if original_img_gray.ndim == 3 and original_img_gray.shape[0] == 1:
        original_img_gray = original_img_gray.squeeze(0)

    # 反归一化
    std = 0.229
    mean = 0.485
    original_img_denorm = (original_img_gray * std + mean) * 255.0
    original_img_denorm = np.clip(original_img_denorm, 0, 255).astype(np.uint8)

    # 打印调试信息

    axes[0].imshow(original_img_denorm, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 异常热力图 - 也需要确保是2D
    ano_map = anomaly_map.squeeze().cpu().numpy()
    if ano_map.ndim == 3 and ano_map.shape[0] == 1:
        ano_map = ano_map.squeeze(0)
    im1 = axes[1].imshow(ano_map, cmap='jet')
    axes[1].set_title('Anomaly Map')
    axes[1].axis('off')

    plt.colorbar(im1, cax=cax)

    # 真实掩码 - 也需要确保是2D
    gt_map = gt_mask.squeeze().cpu().numpy()
    if gt_map.ndim == 3 and gt_map.shape[0] == 1:
        gt_map = gt_map.squeeze(0)
    axes[2].imshow(gt_map, cmap='gray')
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')

    # 叠加显示
    axes[3].imshow(original_img_denorm, cmap='gray')
    axes[3].imshow(ano_map, cmap='jet', alpha=0.5)
    axes[3].set_title('Overlay')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluation_batch_with_visualization(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None,
                                        save_dir="./visualizations", max_samples=20, save_only_anomaly=True):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    os.makedirs(save_dir, exist_ok=True)
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (img, gt, label, img_path) in enumerate(tqdm(dataloader, ncols=80)):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]

            # 生成异常图时使用原始图像大小
            input_size = img.shape[-2:]  # 获取输入图像的高和宽
            anomaly_map, _ = cal_anomaly_maps(en, de, input_size)

            # 统一调整所有图像到相同大小
            if resize_mask is not None:
                # 确保所有图像都调整到相同大小
                target_size = (resize_mask, resize_mask) if isinstance(resize_mask, int) else resize_mask

                anomaly_map = F.interpolate(anomaly_map, size=target_size, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=target_size, mode='nearest')
                # 如果需要，也可以调整原始图像用于显示
                img_display = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
            else:
                img_display = img
                target_size = input_size

            anomaly_map_smoothed = gaussian_kernel(anomaly_map)

            gt_processed = gt.clone()
            gt_processed[gt_processed > 0.5] = 1
            gt_processed[gt_processed <= 0.5] = 0
            if gt_processed.shape[1] > 1:
                gt_processed = torch.max(gt_processed, dim=1, keepdim=True)[0]

            # 保存可视化结果：根据条件保存
            if sample_count < max_samples:
                for i in range(img.shape[0]):
                    if sample_count >= max_samples:
                        break

                    # 如果只保存异常样本，检查标签
                    if save_only_anomaly and label[i].item() == 0:  # 正常样本跳过
                        continue

                    save_path = os.path.join(save_dir, f"sample_{sample_count:04d}_label{label[i].item()}.png")


                    save_comparison_visualization(
                        anomaly_map_smoothed[i],
                        gt_processed[i],
                        img_display[i],  # 使用调整后的图像
                        save_path
                    )
                    sample_count += 1


            # 收集评估数据
            gt_list_px.append(gt_processed)
            pr_list_px.append(anomaly_map_smoothed)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map_smoothed.flatten(1), dim=1)[0]
            else:
                anomaly_map_flat = anomaly_map_smoothed.flatten(1)
                sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:,
                           :int(anomaly_map_flat.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

    gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
    pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
    gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
    pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(
        pr_list_px, pr_list_sp, gt_list_px, gt_list_sp
    )

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]
