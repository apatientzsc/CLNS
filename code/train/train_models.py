import logging
import time
from code import data_process
from sklearn import metrics
import torch
import torch.utils.data as data
from code.data_process import IterLoader
from code.tools import trans
from code.tools.output.log_utils import show_dateset_info, save_model
from code.tools.utils.train_utils import map_crosses, mask_outlier, rename, extract_features, get_proxy
from code.soft_loss import *
from code.correct import add_corr
from code.tree_filter import *
from sklearn import metrics
import logging
import pandas as pd
from pathlib import Path
import os
import pandas as pd


def extract_and_cluster(mode, epoch, args, trainer, cluster):
    kind = 'RGB' if mode == 1 else 'IR'
    features, pseudo_labels, num_clusters, cams, n_class, n_instance, images, real_labels = extract_features(mode, args, trainer, cluster)
    score = metrics.adjusted_rand_score(real_labels, pseudo_labels)

    new_pseudo_labels = pseudo_labels.clone()

    if args.use_tree:
        # 执行过滤
        with torch.no_grad():
            new_pseudo_labels = cam_prototype_calibration(features, pseudo_labels, cams, k_neighbors=50, confidence_threshold=0.8)
    
    #if epoch ==78:
        # Save feature to CSV
        #save_features_to_csv(features, real_labels, cams, mode, epoch, args.dataset)


    # 统一文件写入（只在任一条件为True时执行）
    if True:
        if True:
            save_dir = 'code/txt'
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f"{mode}.txt")

            # 尝试读取已有内容
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
            except FileNotFoundError:
                # 预分配 8 行：ARI、AMI、FMI、V-measure（各 baseline / new）
                lines = ['\n'] * 8

            # ====== 计算所有指标：baseline ======
            ARI = metrics.adjusted_rand_score(real_labels, pseudo_labels)
            AMI = metrics.adjusted_mutual_info_score(real_labels, pseudo_labels)
            FMI = metrics.fowlkes_mallows_score(real_labels, pseudo_labels)
            hom, com, V = metrics.homogeneity_completeness_v_measure(real_labels, pseudo_labels)

            # ====== 计算所有指标：new ======
            ARI_new = metrics.adjusted_rand_score(real_labels, new_pseudo_labels)
            AMI_new = metrics.adjusted_mutual_info_score(real_labels, new_pseudo_labels)
            FMI_new = metrics.fowlkes_mallows_score(real_labels, new_pseudo_labels)
            hom_new, com_new, V_new = metrics.homogeneity_completeness_v_measure(real_labels, new_pseudo_labels)

            # 格式化显示
            def fmt(x):
                return f"{x * 100:.4f}"

            metric_values = [
                fmt(ARI), fmt(ARI_new),
                fmt(AMI), fmt(AMI_new),
                fmt(FMI), fmt(FMI_new),
                fmt(V), fmt(V_new)
            ]

            # 确保行数够
            while len(lines) < 8:
                lines.append('\n')

            # 覆写每一行（按顺序追加数值）
            for i in range(8):
                previous = lines[i].strip()
                new_text = metric_values[i]
                # 追加式写法：之前有内容就拼接
                lines[i] = (previous + ' ' + new_text + '\n') if previous else (new_text + '\n')

            # 写回文件
            with open(filename, 'w') as f:
                f.writelines(lines)

            # cluster 数量
            num_clusters = len(set(new_pseudo_labels)) - (1 if -1 in new_pseudo_labels else 0)



    proxy_labels, num_proxies = get_proxy(new_pseudo_labels, cams)
    logging.info(f"Epoch[{epoch}]==> Create {kind} cluster {num_clusters} classes, proxy {num_proxies} kinds")

    info = [features, new_pseudo_labels, proxy_labels, cams, images, n_class, n_instance, real_labels]
    return info




def save_features_to_csv(features, real_labels, cams, mode, epoch, dataset):
    """
    Save features for samples with specific real labels to a CSV file.

    Args:
        features (torch.Tensor): Feature vectors of shape [N, 2048].
        real_labels (torch.Tensor): Real labels for each sample, shape [N].
        mode (int): Modality (1 for RGB, 0 for IR).
        epoch (int): Current training epoch.
        save_dir (str): Directory to save the CSV file.

    Returns:
        None
    """

    save_dir = os.path.join('/home/xrs/zsc/CLNS_ee/code/csv', str(dataset))
    kind = 'RGB' if mode == 1 else 'IR'
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'{kind}_{epoch}.csv')

    # Filter samples with real_labels in [10, 20, ..., 100]
    target_labels = list(range(10, 301, 1))
    mask = torch.isin(real_labels, torch.tensor(target_labels, device=real_labels.device))
    selected_features = features[mask].cpu().numpy()  # [N_selected, 2048]
    selected_real_labels = real_labels[mask].cpu().numpy()  # [N_selected]
    selected_cams = cams[mask].cpu().numpy()
    selected_mode = [mode] * len(selected_real_labels)  # [N_selected]

    # Create and save DataFrame
    if len(selected_features) > 0:
        feature_columns = [f'feature_{i}' for i in range(2048)]
        data = {
            'real_label': selected_real_labels,
            'cam': selected_cams,
            'mode': selected_mode,
            **{f'feature_{i}': selected_features[:, i] for i in range(2048)}
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f'Saved {len(selected_features)} feature vectors to {filename}')
    else:
        logging.info(f'No features saved for mode {kind} at epoch {epoch} (no samples with target labels)')



def fitter(rgb_info, ir_info):
    [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb, real_labels_rgb] = rgb_info
    [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir, real_labels_ir] = ir_info

    #mask过滤异常值
    mask_rgb = mask_outlier(proxy_labels_rgb)
    mask_ir = mask_outlier(proxy_labels_ir)

    real_labels_rgb, real_labels_ir = rename(real_labels_rgb[mask_rgb]), rename(real_labels_ir[mask_ir])
    proxy_labels_rgb, proxy_labels_ir = rename(proxy_labels_rgb[mask_rgb]), rename(proxy_labels_ir[mask_ir])
    pseudo_labels_rgb, pseudo_labels_ir = rename(pseudo_labels_rgb[mask_rgb]), rename(pseudo_labels_ir[mask_ir])

    cams_rgb, cams_ir = cams_rgb[mask_rgb], cams_ir[mask_ir]
    features_rgb, features_ir = features_rgb[mask_rgb], features_ir[mask_ir]
    images_rgb, images_ir = images_rgb[mask_rgb], images_ir[mask_ir]

    del mask_rgb, mask_ir

    rgb_info = [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb, real_labels_rgb]
    ir_info = [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir, real_labels_ir]

    return rgb_info,ir_info


def add_correlation_weight(rgb_info, ir_info, neighbour, coe, r):
    [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb,
     real_labels_rgb] = rgb_info
    [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir,
     real_labels_ir] = ir_info
    proxy_cluster_features_rgb, proxy_cluster_nums_rgb = generate_cluster_features(proxy_labels_rgb, features_rgb)
    proxy_cluster_features_ir, proxy_cluster_nums_ir = generate_cluster_features(proxy_labels_ir, features_rgb)
    a = 1
    if a == 1:
        correlations_rgb_all, weights_rgb_all = generate_correlations(features_rgb, proxy_labels_rgb,
                                                                      proxy_cluster_nums_rgb,
                                                                      neighbour, coe)
        correlations_ir_all, weights_ir_all = generate_correlations(features_ir, proxy_labels_ir,
                                                                    proxy_cluster_nums_ir,
                                                                    neighbour, coe)

    else:
        correlations_rgb_all, weights_rgb_all = new_generate_correlations(features_rgb, proxy_labels_rgb,
                                                                          proxy_cluster_nums_rgb,
                                                                          r, coe)
        correlations_ir_all, weights_ir_all = new_generate_correlations(features_ir, proxy_labels_ir,
                                                                        proxy_cluster_nums_ir,
                                                                        r, coe)


    rgb_info = [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb,
     real_labels_rgb, correlations_rgb_all, weights_rgb_all]
    ir_info = [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir,
     real_labels_ir, correlations_ir_all, weights_ir_all]

    return rgb_info, ir_info

def add_cross_correlation_weight(rgb_info, ir_info, cross_r2i, cross_i2r, neighbour, coe, r):
    [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb,
                real_labels_rgb, correlations_rgb_all, weights_rgb_all] = rgb_info
    [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir,
               real_labels_ir, correlations_ir_all, weights_ir_all] = ir_info

    proxy_cluster_features_rgb, proxy_cluster_nums_rgb = generate_cluster_features(proxy_labels_rgb, features_rgb)
    proxy_cluster_features_ir, proxy_cluster_nums_ir = generate_cluster_features(proxy_labels_ir, features_rgb)

    a = 1
    if a == 1:
        correlations_rgb_all_cross, weights_rgb_all_cross = generate_correlations_cross(features_rgb, features_ir,
                                                                                        pseudo_labels_ir,
                                                                                        proxy_cluster_nums_ir,
                                                                                        neighbour, pseudo_labels_rgb,
                                                                                        cross_r2i, coe)
        correlations_ir_all_cross, weights_ir_all_cross = generate_correlations_cross(features_ir, features_rgb,
                                                                                      pseudo_labels_rgb,
                                                                                      proxy_cluster_nums_rgb,
                                                                                      neighbour, pseudo_labels_ir,
                                                                                      cross_i2r, coe)

    else:
        correlations_rgb_all_cross, weights_rgb_all_cross = new_generate_correlations_cross(features_rgb, features_ir,
                                                                                            pseudo_labels_ir,
                                                                                            proxy_cluster_nums_ir,
                                                                                            r, pseudo_labels_rgb,
                                                                                            cross_r2i, coe)
        correlations_ir_all_cross, weights_ir_all_cross = new_generate_correlations_cross(features_ir, features_rgb,
                                                                                          pseudo_labels_rgb,
                                                                                          proxy_cluster_nums_rgb,
                                                                                          r, pseudo_labels_ir,
                                                                                          cross_i2r, coe)


    rgb_info = [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb,
                real_labels_rgb, correlations_rgb_all, weights_rgb_all, correlations_rgb_all_cross, weights_rgb_all_cross]
    ir_info = [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir,
               real_labels_ir, correlations_ir_all, weights_ir_all, correlations_ir_all_cross, weights_ir_all_cross]

    return rgb_info, ir_info




def add_fake_cross_correlation_weight(rgb_info, ir_info, cross_r2i, cross_i2r, neighbour, coe):
    [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb,
                real_labels_rgb, correlations_rgb_all, weights_rgb_all] = rgb_info
    [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir,
               real_labels_ir, correlations_ir_all, weights_ir_all] = ir_info

    proxy_cluster_features_rgb, proxy_cluster_nums_rgb = generate_cluster_features(proxy_labels_rgb, features_rgb)
    proxy_cluster_features_ir, proxy_cluster_nums_ir = generate_cluster_features(proxy_labels_ir, features_rgb)

    correlations_rgb_all_cross, weights_rgb_all_cross = fake_generate_correlations_cross(features_rgb, features_ir,
                                                                                    pseudo_labels_ir, proxy_cluster_nums_ir,
                                                                                    neighbour, pseudo_labels_rgb,
                                                                                    cross_r2i, coe)
    correlations_ir_all_cross, weights_ir_all_cross = fake_generate_correlations_cross(features_ir, features_rgb,
                                                                                  pseudo_labels_rgb, proxy_cluster_nums_rgb,
                                                                                  neighbour, pseudo_labels_ir, cross_i2r, coe)

    rgb_info = [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb,
                real_labels_rgb, correlations_rgb_all, weights_rgb_all, correlations_rgb_all_cross, weights_rgb_all_cross]
    ir_info = [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir,
               real_labels_ir, correlations_ir_all, weights_ir_all, correlations_ir_all_cross, weights_ir_all_cross]

    return rgb_info, ir_info





def prepare_datasets(args, rgb_info, ir_info, memory):
    [features_rgb, pseudo_labels_rgb, proxy_labels_rgb, cams_rgb, images_rgb, n_class_rgb, n_instance_rgb, real_labels_rgb, correlations_rgb_all, weights_rgb_all, correlations_rgb_all_cross, weights_rgb_all_cross] = rgb_info
    [features_ir, pseudo_labels_ir, proxy_labels_ir, cams_ir, images_ir, n_class_ir, n_instance_ir, real_labels_ir, correlations_ir_all, weights_ir_all, correlations_ir_all_cross, weights_ir_all_cross] = ir_info

    transform_train_rgb_1, transform_train_rgb_2, transform_train_ir = trans.create_transform(args.dataset, args.data_enhancement)

    magnification = 2 if transform_train_rgb_2 else 1

    i2r_proxy, r2i_proxy, i2r_label, r2i_label = None, None, None, None
    if len(memory.i2r_proxy) != 0 and len(memory.i2r_proxy) != 0:
        i2r_proxy, r2i_proxy = map_crosses(proxy_labels_ir, memory.i2r_proxy), map_crosses(proxy_labels_rgb, memory.r2i_proxy)
        i2r_label, r2i_label = map_crosses(i2r_proxy, memory.rgb_memory.proxy2label), map_crosses(r2i_proxy, memory.ir_memory.proxy2label)

    i2r_class, r2i_class = None, None
    if len(memory.i2r_label) != 0 and len(memory.r2i_label) != 0:
        i2r_class, r2i_class = map_crosses(pseudo_labels_ir, memory.i2r_label), map_crosses(pseudo_labels_rgb, memory.r2i_label)

        # 计算跨模态 ARI 并保存到两个独立的 txt 文件
        if i2r_label is not None and r2i_label is not None:
            # IR 到 RGB 的 ARI：比较 i2r_label（IR 样本的 RGB 伪标签）与 real_labels_rgb
            ari_i2r = metrics.adjusted_rand_score(real_labels_ir, i2r_label)
            # RGB 到 IR 的 ARI：比较 r2i_label（RGB 样本的 IR 伪标签）与 real_labels_ir
            ari_r2i = metrics.adjusted_rand_score(real_labels_rgb, r2i_label)
            # 记录 ARI 分数到日志
            logging.info(f"Cross-modal ARI: IR->RGB = {ari_i2r * 100:.4f}, RGB->IR = {ari_r2i * 100:.4f}")

            # 保存 ARI 分数到两个独立的 txt 文件，仅写入数值，以空格分隔
            save_dir = 'code/txt'
            os.makedirs(save_dir, exist_ok=True)

            # 保存 IR->RGB ARI
            ir2rgb_filename = os.path.join(save_dir, 'ir2rgb_ari.txt')
            with open(ir2rgb_filename, 'a') as f:
                f.write(f"{ari_i2r * 100:.4f} ")

            # 保存 RGB->IR ARI
            rgb2ir_filename = os.path.join(save_dir, 'rgb2ir_ari.txt')
            with open(rgb2ir_filename, 'a') as f:
                f.write(f"{ari_r2i * 100:.4f} ")


    train_dataset_rgb = data_process.create(
        'train',
        train_image=images_rgb, proxy=proxy_labels_rgb, label=pseudo_labels_rgb, cam=cams_rgb, cross_proxy=r2i_proxy, cross_label=r2i_label, cross_class=r2i_class,
        correlations=correlations_rgb_all, weights=weights_rgb_all,
        correlations_cross=correlations_rgb_all_cross, weights_cross=weights_rgb_all_cross,
        transform_1=transform_train_rgb_1, transform_2=transform_train_rgb_2
    )
    train_dataset_ir = data_process.create(
        'train',
        train_image=images_ir, proxy=proxy_labels_ir, label=pseudo_labels_ir, cam=cams_ir, cross_proxy=i2r_proxy, cross_label=i2r_label, cross_class=i2r_class,
        correlations=correlations_ir_all, weights=weights_ir_all,
        correlations_cross=correlations_ir_all_cross, weights_cross=weights_ir_all_cross,
        transform_1=transform_train_ir
    )

    show_dateset_info(
        n_class_rgb, n_class_ir, n_instance_rgb, n_instance_ir,
        len(torch.unique(pseudo_labels_rgb)), len(torch.unique(pseudo_labels_ir)), len(train_dataset_rgb), len(train_dataset_ir),
    )

    return train_dataset_rgb, train_dataset_ir, magnification

def create_dataloaders(train_dataset_rgb, train_dataset_ir, args, rgb_sampler, ir_sampler, magnification = 1):
    batch_size = args.train_batch_size
    rgb_trainloader = IterLoader(
        data.DataLoader(train_dataset_rgb, batch_size=batch_size, sampler=rgb_sampler,
                        num_workers=args.train_num_workers,
                        drop_last=True)
    )


    ir_trainloader = IterLoader(
        data.DataLoader(train_dataset_ir, batch_size=batch_size * magnification, sampler=ir_sampler,
                        num_workers=args.train_num_workers,
                        drop_last=True)
    )


    return rgb_trainloader, ir_trainloader


def valid(args, trainer, epoch, best_cmc):
    start = time.time()
    logging.info(f"Epoch[{epoch}] Test start")
    cmc_rank_1 = trainer.valid(args, args.test_mode_1, args.mode_1)
    cmc_rank_2 = trainer.valid(args, args.test_mode_2, args.mode_2)
    mean_cmc = (cmc_rank_1+cmc_rank_2) / 2
    if mean_cmc > best_cmc:
        logging.info(
            f"Epoch [{epoch}], save better model Rank-1: {mean_cmc:.2%} to replace original model Rank-1: {best_cmc:.2%}")
        best_cmc = mean_cmc
        save_model(args, trainer, epoch, 0)

    logging.info(f"Epoch [{epoch}] Test end, time cost {time.time() - start}")
    return best_cmc

def save(args, trainer, epoch):
    logging.info(f"Epoch [{epoch}], save model for record")
    save_model(args, trainer, epoch, 1)