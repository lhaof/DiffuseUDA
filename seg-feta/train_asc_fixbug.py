import argparse
import logging
import torch
import shutil
import time
import torch.nn as nn
import os
import random
import numpy as np
import torchmetrics
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import ramps
from dataloader_seg import BaseDataSets
from segresnet import SegResNet

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str,
                    default='_stm', help='Dataset type')
parser.add_argument('--root_path_t', type=str,
                    default='./seg_dataset/feta', help='Name of Experiment')
parser.add_argument('--root_path_s', type=str,
                    default='./seg_dataset/atlas', help='Name of Experiment')
parser.add_argument('--train_data_s', type=str,
                    default='train-old.list', help='Name of Dataset')
parser.add_argument('--train_data_t', type=str,
                    default='train.list', help='Name of Dataset')
parser.add_argument('--test_data', type=str,
                    default='test.list', help='Name of Dataset')
parser.add_argument('--exp', type=str,
                    default='ASC', help='Name of Experiment')
parser.add_argument('--num_classes', type=int,  default=8,
                    help='output channel of network')
parser.add_argument('--max_epoch', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--patch_size', type=list,  default=[128,128,128],
                    help='patch size of network input')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--epoch_gap', type=int, default=5,
                    help='choose epoch gap to val model')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=200, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=100.0, help='consistency_rampup')
parser.add_argument('--zoom', type=int, default=1,
                    help='whether use zoom training')
parser.add_argument('--crop', type=int, default=1,
                    help='whether use crop training')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def calculate_dice(output, target, num_classes=8):
    dice_scores = torchmetrics.functional.dice(torch.from_numpy(output).cuda(), torch.from_numpy(target).cuda(), average='none', num_classes=num_classes)
    return dice_scores[1:]  # Skip the background class

def evaluate_model(model, valid_loader):
    """
    评估模型在验证集上的性能
    """
    model.eval()
    total_dice = 0.0
    num_batches = len(valid_loader)

    for i_batch, sampled_batch in enumerate(valid_loader):
        volume_batch, label = sampled_batch['image'], sampled_batch['mask']
        label = label.squeeze(1).squeeze(0).cpu().detach().numpy()
        
        with torch.no_grad():
            output = model(volume_batch.cuda())['seg']
            output = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
        
        dice_scores = calculate_dice(output, label)
        total_dice += dice_scores.mean().item()

    average_dice = total_dice / num_batches
    return round(average_dice, 4)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def extract_ampl_phase(fft_im):
    """
    Extract the amplitude and phase from the FFT result.
    """
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    """
    Replace the low-frequency amplitude of the source image with that of the target image.
    """
    bs, n, h, w, d = amp_src.size()
    d *= 2  # Restore full depth
    b = int(np.floor(0.5 * np.amin((h, w, d)) * L))  # Calculate the size of the low-frequency region
    
    if b > 0:
        # Replace the low-frequency amplitude region
        amp_src[:, :, :b, :b, :b] = amp_trg[:, :, :b, :b, :b]  # Top-left
        amp_src[:, :, -b:, :b, :b] = amp_trg[:, :, -b:, :b, :b]  # Bottom-left
        amp_src[:, :, :b, -b:, :b] = amp_trg[:, :, :b, -b:, :b]  # Top-right
        amp_src[:, :, -b:, -b:, :b] = amp_trg[:, :, -b:, -b:, :b]  # Bottom-right
    
    return amp_src

def calculate_L_based_on_histogram(src_img, trg_img, bins=256):
    """
    Calculate the parameter L based on the histogram difference between source and target images.
    """
    def calculate_histogram(image, bins):
        """
        Calculate the histogram of a 3D image.
        """
        histogram, _ = np.histogram(image.flatten(), bins=bins, range=[0, bins])
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + 1e-7)  # Normalize
        return histogram
    
    src_img_np = src_img.cpu().detach().numpy()*255
    trg_img_np = trg_img.cpu().detach().numpy()*255
    
    # Calculate histograms
    hist_src = calculate_histogram(src_img_np, bins)
    hist_trg = calculate_histogram(trg_img_np, bins)
    
    # Calculate histogram difference
    hist_diff = np.sum((hist_src - hist_trg) ** 2)
    
    # Normalize histogram difference to determine L
    L = hist_diff / (hist_diff + 1)  # Ensure L is between 0 and 1
    
    return L

def FDA_source_to_target(src_img, trg_img, L=None):
    """
    Fuse the content of the source image with the style of the target image.
    """
    if L is None:
        L = calculate_L_based_on_histogram(src_img, trg_img)
        L = max(L, 0.1)
        uncertainty_factor = random.uniform(0.5, 2)
        L = L * uncertainty_factor
        print("Calculated L:", L)

    # Compute the FFT of the source and target images
    fft_src = torch.fft.rfftn(src_img, dim=(-3, -2, -1))
    fft_trg = torch.fft.rfftn(trg_img, dim=(-3, -2, -1))

    # Extract amplitude and phase
    amp_src, pha_src = extract_ampl_phase(fft_src)
    amp_trg, _ = extract_ampl_phase(fft_trg)

    # Replace low-frequency amplitude
    amp_src = low_freq_mutate(amp_src, amp_trg, L=L)

    # Reconstruct the FFT result
    fft_src_ = torch.polar(amp_src, pha_src)

    # Perform inverse FFT to get the fused image
    src_in_trg = torch.fft.irfftn(fft_src_, dim=(-3, -2, -1), s=src_img.shape[-3:])

    return src_in_trg

class BoxMaskGenerator:
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=False, prop_by_area=True, within_bounds=True, invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        if rng is None:
            rng = np.random

        mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
        zero_mask = mask_props == 0.0

        if self.random_aspect_ratio:
            y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
            x_props = mask_props / y_props
        else:
            z_props = y_props = x_props = np.sqrt(mask_props)
        
        fac = np.sqrt(1.0 / self.n_boxes)
        z_props *= fac
        y_props *= fac
        x_props *= fac

        z_props[zero_mask] = 0
        y_props[zero_mask] = 0
        x_props[zero_mask] = 0

        sizes = np.round(np.stack([z_props, y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        masks = np.zeros((n_masks, 1) + mask_shape) if self.invert else np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for z0, y0, x0, z1, y1, x1 in sample_rectangles:
                masks[i, 0, int(z0):int(z1), int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(z0):int(z1), int(y0):int(y1), int(x0):int(x1)]
        return masks

    def append_to_batch(self, *batch):
        x = batch[0]
        params = self.generate_params(len(x), x.shape[2:4])
        return batch + (params,)


def train(args):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    save_path = 'runs/' + 'asc_fixbug_2var' + '/'
    os.makedirs(save_path, exist_ok=True)

    def create_model(ema=False):
        # Network definition
        model = SegResNet(in_channels=1, out_channels=args.num_classes, init_filters=32).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = nn.DataParallel(create_model())
    ema_model = nn.DataParallel(create_model(ema=True))

    print(args.seed)

    batch_size_half = int(batch_size / 2)
    train_seg = BaseDataSets(data_dir=args.root_path_s, mode="train", list_name=args.train_data_s,
                             patch_size=args.patch_size, crop=args.crop, zoom=args.zoom, transform=None, type=args.type)
    valid_seg = BaseDataSets(data_dir=args.root_path_t, mode="test", list_name=args.train_data_t,
                             patch_size=args.patch_size, crop=args.crop, zoom=args.zoom, transform=None, type=args.type)
    test_seg = BaseDataSets(data_dir=args.root_path_t, mode="test", list_name=args.test_data,
                            patch_size=args.patch_size, crop=args.crop, zoom=args.zoom, type=args.type)
    train_loader_seg = DataLoader(train_seg, batch_size=int(args.batch_size / 2), shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader_seg = DataLoader(valid_seg, batch_size=int(args.batch_size / 2), shuffle=False, num_workers=4, pin_memory=True)
    valid_loader_seg2 = DataLoader(valid_seg, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    test_loader_seg = DataLoader(test_seg, batch_size=1, shuffle=False, num_workers=1)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    dice_loss = DiceLoss(num_classes)
    mse_loss = nn.MSELoss()

    logging.info("{} iterations per epoch".format(len(valid_loader_seg)))
    iter_num = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    max_iterations = max_epoch * len(valid_loader_seg)

    for epoch_num in iterator:
        # 缓存 valid_loader_seg 的所有批次并洗牌
        valid_batches = list(valid_loader_seg)
        random.shuffle(valid_batches)
        valid_batch_idx = 0

        for i_batch, sampled_batch_s in enumerate(train_loader_seg):
            # 使用伪随机的方法，确保每个 valid_loader_seg 的样本都至少出现一次
            if valid_batch_idx >= len(valid_batches):
                random.shuffle(valid_batches)
                valid_batch_idx = 0

            sampled_batch_t = valid_batches[valid_batch_idx]
            valid_batch_idx += 1

            volume_batch = torch.cat((sampled_batch_s['image'], sampled_batch_t['image']))
            label_batch_s = sampled_batch_s['mask']

            volume_batch = volume_batch.cuda()
            label_batch_s = label_batch_s.cuda()

            volume_batch_t = volume_batch[batch_size_half:]
            volume_batch_s = volume_batch[:batch_size_half]

            # Supervised loss
            outputs_soft_s = torch.softmax(model(volume_batch_s)['seg'], dim=1)
            src_in_trg = FDA_source_to_target(src_img=volume_batch_s, trg_img=volume_batch_t, L=None)
            outputs_soft_sft = torch.softmax(model(src_in_trg)['seg'], dim=1)
            loss_sup = dice_loss(outputs_soft_s, label_batch_s)
            loss_supft = dice_loss(outputs_soft_sft, label_batch_s)

            # Inter-domain
            trg_in_src = FDA_source_to_target(src_img=volume_batch_t, trg_img=volume_batch_s, L=None)
            mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5), random_aspect_ratio=False, n_boxes=1, invert=True)
            cut_mask = mask_generator.generate_params(int(batch_size_half / 2), (volume_batch_t.shape[-3], volume_batch_t.shape[-2], volume_batch_t.shape[-1]))
            mask = torch.tensor(cut_mask).type(torch.FloatTensor).cuda()
            volume_batch_t0 = volume_batch_t[0:batch_size_half // 2, ...]
            volume_batch_t1 = volume_batch_t[batch_size_half // 2:, ...]
            batch_tx_mixed = volume_batch_t0 * (1.0 - mask) + volume_batch_t1 * mask
            outputs_soft_tx = torch.softmax(model(batch_tx_mixed)['seg'], dim=1)
            batch_tfs_mixed = trg_in_src[:1] * (1.0 - mask) + trg_in_src[1:] * mask
            outputs_soft_tfs = torch.softmax(model(batch_tfs_mixed)['seg'], dim=1)

            # Domain consistency loss
            with torch.no_grad():
                ema_output = torch.softmax(ema_model(volume_batch_t)['seg'], dim=1)
                ema_outputfs = torch.softmax(ema_model(trg_in_src)['seg'], dim=1)
                ema_output_t0, ema_output_t1 = ema_output[:batch_size_half // 2], ema_output[batch_size_half // 2:]
                batch_pred_mixed = ema_output_t0 * (1.0 - mask) + ema_output_t1 * mask
                ema_output_tfs0, ema_output_tfs1 = ema_outputfs[:batch_size_half // 2], ema_outputfs[batch_size_half // 2:]
                batch_pred_mixed2 = ema_output_tfs0 * (1.0 - mask) + ema_output_tfs1 * mask
            inter_loss = mse_loss(outputs_soft_tfs, batch_pred_mixed) + mse_loss(outputs_soft_tx, batch_pred_mixed2)
            consistency_weight = get_current_consistency_weight(epoch_num)

            # Total loss
            loss = loss_sup + loss_supft + consistency_weight * inter_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - (iter_num + 1) / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            logging.info(
                'iteration %d : loss : %f loss_sup : %f loss_supft : %f loss_inter : %f' %
                (iter_num, loss.item(), loss_sup.item(), loss_supft.item(), inter_loss))

        with open(save_path + '/results.txt', 'a') as f:
            f.write("epoch:" + str(epoch_num) + '\n')
        valid_dice = evaluate_model(model, valid_loader_seg2)
        test_dice = evaluate_model(model, test_loader_seg)
        weights_path = save_path + str(epoch_num) + '-valid-' + str(valid_dice) + '-test-' + str(test_dice) + '.pth'
        torch.save(model.state_dict(), weights_path)
        print('Valid dice score:', valid_dice)
        print('Test dice score:', test_dice)

    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_name = args.exp + '.py'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    logfile = './log.txt'
    fh = logging.FileHandler(logfile, mode='a')  # open的打开模式这里可以进行参考
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)  # 输出到console的log等级的开关
    # 第四步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(str(args))
    train(args)


