import os
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from skimage import morphology, measure

from utils import read_list, maybe_mkdir, test_single_case, read_data, config as config_module
from segresnet import SegResNet
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-t', '--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='fully')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--speed', type=int, default=0)
parser.add_argument('-g', '--gpu', type=str, default='1')
args = parser.parse_args()
# Set GPU environment
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Configuration setup
config = config_module.Config(args.task)

def keep_largest_component(segmentation, num_classes):
    for class_id in range(1, num_classes):  # Assuming class 0 is the background
        class_mask = segmentation == class_id
        labeled_mask = measure.label(class_mask)
        if np.max(labeled_mask) > 0:  # If there are any objects
            regions = measure.regionprops(labeled_mask)
            largest_region = max(regions, key=lambda r: r.area)
            new_mask = (labeled_mask == largest_region.label)
            segmentation[class_mask] = 0
            segmentation[new_mask] = class_id
    return segmentation

if __name__ == '__main__':
    stride_dict = {
        0: (16, 4),
        1: (64, 16),
        2: (128, 32),
    }
    stride = stride_dict[args.speed]

    if 'mr2ct' in args.task:
        test_save_path = f'./logs/{args.exp}/mr2ct/'
        maybe_mkdir(test_save_path)
        print(test_save_path)

        ckpt_paths = [
            './logs/Exp_UDA_MMWHS_mr2ct_ASCP/asc_crop1/fold1/ckpts/101-valid-0.8330137-test-.pth',
            './logs/Exp_UDA_MMWHS_mr2ct_ASCP/asc_crop1/fold1/ckpts/109-valid-0.84361154-test-.pth',
            './logs/Exp_UDA_MMWHS_mr2ct_ASCP/asc_crop1/fold1/ckpts/102-valid-0.84255075-test-.pth',
        ]
    else:
        test_save_path = f'./logs/{args.exp}/ct2mr/'
        maybe_mkdir(test_save_path)
        print(test_save_path)

        ckpt_paths = [
            # './logs/Exp_UDA_MMWHS_ct2mr_ASCP/asc_crop1/fold1/ckpts/185-valid-0.80432016-test-.pth',
            # './logs/Exp_UDA_MMWHS_ct2mr_ASCP/asc_crop1/fold1/ckpts/193-valid-0.8053851-test-.pth',
            # './logs/Exp_UDA_MMWHS_ct2mr_ASCP/asc_crop1/fold1/ckpts/187-valid-0.80467695-test-.pth',
            '/mntcephfs/lab_data/wangyitao/GenericSSL/logs/Exp_UDA_MMWHS_ct2mr_DiffUDA/asc_crop1/fold1/ckpts/best_model.pth',
            '/mntcephfs/lab_data/wangyitao/GenericSSL/logs/Exp_UDA_MMWHS_ct2mr_DiffUDA/asc_crop2/fold1/ckpts/best_model.pth',
            '/mntcephfs/lab_data/wangyitao/GenericSSL/logs/Exp_UDA_MMWHS_ct2mr_DiffUDA/asc_crop3/fold1/ckpts/best_model.pth',
        ]

    models = []
    for ckpt_path in ckpt_paths:
        model = SegResNet(in_channels=1, out_channels=config.num_cls, init_filters=config.n_filters)
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        model.cuda()
        model.eval()
        models.append(model)
        print(f'Loaded checkpoint from {ckpt_path}')

    ids_list = read_list(args.split, task=args.task)
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task=args.task, normalize=True)
        probability_maps = []

        for model in models:
            with torch.no_grad():
                _, probabilities = test_single_case(
                    model,
                    image,
                    stride[0],
                    stride[1],
                    config.patch_size,
                    num_classes=config.num_cls
                )
                probability_maps.append(probabilities)
        
        # Average predictions from all models and round to the nearest integer
        ensemble_pred = np.mean(np.array(probability_maps), axis=0)
        ensemble_pred = np.argmax(ensemble_pred, axis=0).astype(int)
        # Post-processing
        final_pred = keep_largest_component(ensemble_pred.astype(int), config.num_cls)

        # Save the result
        out = sitk.GetImageFromArray(final_pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')
