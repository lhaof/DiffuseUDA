import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='synapse', required=True)
parser.add_argument('--exp', type=str, default='diffusion')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, required=True)
parser.add_argument('-su', '--split_unlabeled', type=str, required=True)
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=300)
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--unsup_loss', type=str, default='w_ce+dice')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--mu', type=float, default=2.0)
parser.add_argument('-s', '--ema_w', type=float, default=0.99)
parser.add_argument('-r', '--mu_rampup', action='store_true', default=True) # <--
parser.add_argument('-cr', '--rampup_epoch', type=float, default=None) # 100
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import ramps
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader,sampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from DiffVNet.diff_vnet import DiffVNet
from segresnet import SegResNet
from utils import EMA, maybe_mkdir, get_lr, fetch_data, GaussianSmoothing, seed_worker, poly_lr, print_func, sigmoid_rampup
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.data_loaders import DatasetAllTasks
from utils.config import Config
from data.StrongAug import get_StrongAug, ToTensor, CenterCrop
from utilsseg import *
config = Config(args.task)
from train_asc import FDA_source_to_target



def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


def make_loader(split, dst_cls=DatasetAllTasks, is_training=True, unlabeled=False, task="", transforms_tr=None,resample=None):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=resample,
            unlabeled=unlabeled,
            transform=transforms_tr,
            task=task,
            num_cls=config.num_cls
        )
        if resample != None:
            batch_sampler = sampler.RandomSampler(data_source= dst ,replacement=True,num_samples=resample)
            return DataLoader(
                dst,
                batch_size=int(config.batch_size/2),
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                drop_last=True,
                sampler=batch_sampler
            )
        else:
            return DataLoader(
                dst,
                batch_size=int(config.batch_size/2),
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                drop_last=True
            )

    else:
        dst = dst_cls(
            split=split,
            is_val=True,
            task=task,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def create_model(ema=False):
    model = SegResNet(in_channels=1, out_channels=config.num_cls, init_filters=config.n_filters).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 200 * ramps.sigmoid_rampup(epoch, 100.0)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

import torch
import torch.nn as nn
from monai.losses import DiceLoss

def compute_dice_ce_loss(pred, ground_truth):
    """
    Computes the combined Dice and Cross-Entropy loss for given raw predictions and ground truth labels.

    Parameters:
    - pred (torch.Tensor): Raw predictions, shape (batch_size, num_classes, D, H, W)
    - ground_truth (torch.Tensor): Ground truth labels, shape (batch_size, 1, D, H, W)

    Returns:
    - total_loss (torch.Tensor): Sum of Dice loss and Cross-Entropy loss
    """
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    ce_out = ce_loss(pred, torch.squeeze(ground_truth, dim=1))
    dice_out = dice_loss(pred, ground_truth)
    return dice_out + ce_out

if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    vis_path = os.path.join(snapshot_path, 'vis')
    maybe_mkdir(vis_path)

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S', force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    transforms_train_labeled = get_StrongAug(config.patch_size, 3, 0.7)
    transforms_train_unlabeled = get_StrongAug(config.patch_size, 3, 0.7)

    if "mmwhs" not in args.task:
        print(args.split_unlabeled,args.split_labeled)
        unlabeled_loader = make_loader(args.split_unlabeled, transforms_tr=transforms_train_unlabeled, task=args.task, unlabeled=True)
        labeled_loader = make_loader(args.split_labeled, transforms_tr=transforms_train_labeled, task=args.task, repeat=len(unlabeled_loader.dataset))
    else:
        print(args.split_unlabeled,args.split_labeled,11111111)
        labeled_loader = make_loader(args.split_labeled, transforms_tr=transforms_train_labeled, task=args.task)
        unlabeled_loader = make_loader(args.split_unlabeled, transforms_tr=transforms_train_unlabeled, task=args.task, unlabeled=True,resample=len(labeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, task=args.task, is_training=False)
    print(len(unlabeled_loader.dataset))


    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )
    # optimizer = optim.AdamW(model.parameters()) # , lr=base_lr

    best_eval = 0.0
    best_epoch = 0
    max_epoch= args.max_epoch
    iterator = tqdm(range(max_epoch), ncols=70)
    batch_size_half = int(config.batch_size / 2)
    mse_loss = nn.MSELoss()
    max_iterations = max_epoch * len(labeled_loader)
    base_lr = args.base_lr
    iter_num = 0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(zip(labeled_loader, unlabeled_loader)):
            sampled_batch_s, sampled_batch_t = sampled_batch[0], sampled_batch[1]
            volume_batch, label_batch_s = torch.cat((sampled_batch_s['image'],sampled_batch_t['image'])), torch.tensor(sampled_batch_s['label'])

            volume_batch, label_batch_s = volume_batch.cuda(), label_batch_s.unsqueeze(dim = 1).cuda()
            volume_batch_t = volume_batch[batch_size_half:]
            volume_batch_s = volume_batch[:batch_size_half]

            # Supervised loss
            outputs_soft_s = model(volume_batch_s.cuda())['seg']
            src_in_trg = FDA_source_to_target(src_img=volume_batch_s.cuda(),trg_img=volume_batch_t.cuda(),L=None)
            outputs_soft_sft = model(src_in_trg)['seg']
            loss_sup = compute_dice_ce_loss(outputs_soft_s, label_batch_s.long())
            loss_supft = compute_dice_ce_loss(outputs_soft_sft, label_batch_s.long())

            # Inter-domain
            trg_in_src = FDA_source_to_target(src_img=volume_batch_t,trg_img=volume_batch_s,L=None)
            mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5), random_aspect_ratio=False,
                                              n_boxes=1, invert=True)
            cut_mask = mask_generator.generate_params(int(batch_size_half / 2),(volume_batch_t.shape[-3],
                                                       volume_batch_t.shape[-2],
                                                       volume_batch_t.shape[-1]))
            mask = torch.tensor(cut_mask).type(torch.FloatTensor).cuda()
            volume_batch_t0 = volume_batch_t[0:batch_size_half // 2, ...]
            volume_batch_t1 = volume_batch_t[batch_size_half // 2:, ...]
            batch_tx_mixed = volume_batch_t0 * (1.0 - mask) + volume_batch_t1 * mask
            outputs_soft_tx = torch.softmax(model(batch_tx_mixed)['seg'], dim=1)
            batch_tfs_mixed = trg_in_src[:1] * (1.0 - mask) + trg_in_src[1:] * mask
            outputs_soft_tfs = torch.softmax(model(batch_tfs_mixed)['seg'], dim=1)

            with torch.no_grad():
                ema_output = torch.softmax(ema_model(volume_batch_t)['seg'], dim=1)
                ema_outputfs = torch.softmax(ema_model(trg_in_src)['seg'], dim=1)
                ema_output_t0, ema_output_t1 = ema_output[:batch_size_half // 2], ema_output[batch_size_half // 2:]
                batch_pred_mixed = ema_output_t0 * (1.0 - mask) + ema_output_t1 * mask
                ema_output_tfs0, ema_output_tfs1 = ema_outputfs[:batch_size_half // 2], ema_outputfs[batch_size_half // 2:]
                batch_pred_mixed2 = ema_output_tfs0 * (1.0 - mask) + ema_output_tfs1 * mask

            inter_loss = mse_loss(outputs_soft_tfs, batch_pred_mixed) + mse_loss(outputs_soft_tx, batch_pred_mixed2)
            consistency_weight = get_current_consistency_weight(epoch_num)

            loss = loss_sup + loss_supft + consistency_weight * inter_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - (iter_num+1) / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            print(
                'iteration %d : loss : %f loss_sup : %f loss_supft : %f loss_inter : %f' %
                (iter_num, loss.item(), loss_sup.item(), loss_supft.item(), inter_loss))
            logging.info(
                'iteration %d : loss : %f loss_sup : %f loss_supft : %f loss_inter : %f' %
                (iter_num, loss.item(), loss_sup.item(), loss_supft.item(), inter_loss))
            loss_list = []
            loss_sup_list= []
            loss_supft_list = []
            inter_loss_list = []
            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_supft_list.append(loss_supft.item())
            inter_loss_list.append(inter_loss.item())

        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/deno', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/unsup', np.mean(inter_loss_list), epoch_num)
        writer.add_scalar('loss/diff', np.mean(loss_supft_list), epoch_num)
        # writer.add_scalars('class_weights', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_diff))), epoch_num)

        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)} | lr : {get_lr(optimizer)}')
        # logging.info(f"     diff_w: {print_func(weight_diff)}")
        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        # =======================================================================================
        # Validation
        # =======================================================================================
        if epoch_num % 1 == 0:

            dice_list = [[] for _ in range(config.num_cls-1)]
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8, do_bg=False)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    p_u_theta = model(image.cuda())['seg']
                    del image

                    shp = (p_u_theta.shape[0], config.num_cls) + p_u_theta.shape[2:]
                    gt = gt.long()

                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    p_u_theta = torch.argmax(p_u_theta, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, p_u_theta, 1)


                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            writer.add_scalar('val_dice', np.mean(dice_mean), epoch_num)
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path_1 = os.path.join(snapshot_path, "ckpts/"+str(epoch_num) + '-valid-' + str(np.mean(dice_mean)) + '-test-' + '.pth')
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                }, save_path_1)
                torch.save({
                    'state_dict': model.state_dict(),
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    writer.close()
