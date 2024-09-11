import argparse
import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
import SimpleITK as sitk
from monai.metrics import SurfaceDiceMetric as NSD
from skimage import morphology
from dataloader_seg import BaseDataSets
from resunet import ResUnet
from segresnet import SegResNet


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def get_points_from_heatmap_torch(heatmap):
    max_indices = torch.argmax(heatmap.view(heatmap.size(0), heatmap.size(1), -1), dim=2)
    z_indices = max_indices % heatmap.size(4)
    y_indices = (max_indices // heatmap.size(4)) % heatmap.size(3)
    x_indices = max_indices // (heatmap.size(3) * heatmap.size(4))
    return torch.stack([x_indices, y_indices, z_indices], dim=2)


def post_processing(segmentation_labels, num_classes=8):
    for class_id in range(num_classes):
        class_mask = segmentation_labels == class_id
        class_mask = morphology.remove_small_objects(class_mask, min_size=50)
        class_mask = morphology.remove_small_holes(class_mask, area_threshold=50)
        segmentation_labels[class_mask] = class_id
    return segmentation_labels


def one_hot_logits(logits):
    _, predicted = torch.max(logits, dim=1, keepdim=True)
    one_hot = torch.zeros_like(logits, dtype=torch.float32)
    return one_hot.scatter_(1, predicted, 1)


def one_hot_encoder(input_tensor, n_classes):
    return torch.stack([input_tensor == i * torch.ones_like(input_tensor) for i in range(n_classes)], dim=1).float()


def _show_nsd(input, target, spacing_mm=None):
    input, target = input.unsqueeze(0), target.unsqueeze(0)
    input = one_hot_logits(one_hot_encoder(input, n_classes=8))
    target = one_hot_encoder(target, n_classes=8)
    compute_nsd = NSD(class_thresholds=[1] * 7, reduction='mean_channel')
    compute_nsd(input, target, spacing=spacing_mm)
    return compute_nsd.aggregate(reduction="none").squeeze()


def load_model(model_name, checkpoint):
    if model_name == 'asc':
        model = SegResNet(in_channels=1, out_channels=8, init_filters=32)
        model = nn.DataParallel(model).cuda()
    elif model_name == 'resunet':
        model = ResUnet(num_classes=8).cuda()
    else:
        raise NotImplementedError
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    return model


def evaluate_models(model_name, checkpoints):
    start_time = time.time()

    current_root = f'./results_feta/fusion_deformable/'
    current_root_img, current_root_mask = os.path.join(current_root, 'img/'), os.path.join(current_root, 'mask/')
    os.makedirs(current_root_img, exist_ok=True)
    os.makedirs(current_root_mask, exist_ok=True)

    models = [load_model(model_name, checkpoint) for checkpoint in checkpoints]

    record_file = open(os.path.join(current_root, 'results.txt'), 'w')
    test_seg = BaseDataSets(data_dir='./seg_dataset/feta', mode="test", list_name='test.list', crop=True, zoom=True)
    test_loader_seg = DataLoader(test_seg, batch_size=1, shuffle=False)

    metric_list, surface_list, normal_value, abnormal_value, normal_surface, abnormal_surface = [], [], [], [], [], []

    for i_batch, sampled_batch in enumerate(test_loader_seg):
        case_start_time = time.time()
        volume_batch, label, ori = sampled_batch['image'], sampled_batch['mask'], sampled_batch['ori']
        spacing = [sampled_batch['spacing'][i].item() for i in range(3)]
        abnormal, ori = sampled_batch['pathological'], [i.item() for i in ori]
        label = label.squeeze(1).cuda().cpu().squeeze(0).detach().numpy()
        
        with torch.no_grad():
            outputs = [model(volume_batch.cuda())['seg'] if "asc" in model_name else model(volume_batch.cuda()) for model in models]
            output_ensemble = torch.mean(torch.stack(outputs, dim=0), dim=0)
            out = torch.argmax(torch.softmax(output_ensemble, dim=1), dim=1).cpu().squeeze(0).detach().numpy()
        
        out = post_processing(out)
        metric_i = torchmetrics.functional.dice(torch.tensor(out), torch.tensor(label), average='none', num_classes=8)[1:].mean().item()
        surface_dice = _show_nsd(torch.from_numpy(out).cuda(), torch.from_numpy(label).cuda(), spacing).mean().item()

        # Convert to percentage and round to two decimal places
        metric_i = round(metric_i * 100, 2)
        surface_dice = round(surface_dice * 100, 2)

        record_file.write(f'{metric_i}%\n{surface_dice}%\n')
        metric_list.append(metric_i)
        surface_list.append(surface_dice)

        value_list = normal_value if abnormal.item() == 0 else abnormal_value
        surface_list = normal_surface if abnormal.item() == 0 else abnormal_surface
        value_list.append(metric_i)
        surface_list.append(surface_dice)

        name = sampled_batch['idx'][0].split('.')[0]
        volume_batch = (volume_batch.squeeze(0).squeeze(0).cpu().detach().numpy() * 255).astype(np.int16)
        label_img, out_img = label.astype(np.int16), out.astype(np.int16)

        for img, path in zip([volume_batch, label_img, out_img], [current_root_img + name + '_srr.nii.gz', current_root + name + 'label.nii.gz', current_root_mask + name + '_parcellation.nii.gz']):
            img = sitk.GetImageFromArray(img)
            img.SetDirection(ori)
            sitk.WriteImage(img, path)

        print(f'case end time: {time.time() - case_start_time}s')

    avg_metric = lambda x: round(sum(x) / len(x), 2)
    dice, surf = [avg_metric(lst) for lst in [metric_list, surface_list]]
    abn_dice, norm_dice = [avg_metric(lst) for lst in [abnormal_value, normal_value]]
    abn_surf, norm_surf = [avg_metric(lst) for lst in [abnormal_surface, normal_surface]]

    record_file.write(f'average dice: {dice}%\n')
    record_file.write(f'average abn dice: {abn_dice}%\n')
    record_file.write(f'average normal dice: {norm_dice}%\n')
    record_file.write(f'average nsd: {surf}%\n')
    record_file.write(f'average abn nsd: {abn_surf}%\n')
    record_file.write(f'average normal nsd: {norm_surf}%\n')

    print(f'infer time: {time.time() - start_time}s')
    print(f'average dice: {dice}%')
    print(f'average abn dice: {abn_dice}%')
    print(f'average normal dice: {norm_dice}%')
    print(f'average nsd: {surf}%')
    print(f'average abn nsd: {abn_surf}%')
    print(f'average normal nsd: {norm_surf}%')


def get_top_checkpoints(folder_path, top_n=5):
    checkpoint_files = glob.glob(os.path.join(folder_path, '*.pth'))
    checkpoints_with_scores = []

    for checkpoint in checkpoint_files:
        try:
            # Assuming filename format is '<epoch>-valid-<val_score>-test-<test_score>.pth'
            parts = os.path.basename(checkpoint).split('-')
            val_score = float(parts[2])
            checkpoints_with_scores.append((val_score, checkpoint))
        except:
            continue

    checkpoints_with_scores.sort(reverse=True, key=lambda x: x[0])
    top_checkpoints = [ckpt for _, ckpt in checkpoints_with_scores[:top_n]]

    print("Top checkpoints selected for evaluation:")
    for ckpt in top_checkpoints:
        print(ckpt)

    return top_checkpoints


if __name__ == "__main__":
    # folder_path = "./runs/asc_v5/"
    folder_path = "./runs/asc_diffuse_x4/"
    top_checkpoints = get_top_checkpoints(folder_path, top_n=1)

    evaluate_models('asc', top_checkpoints)