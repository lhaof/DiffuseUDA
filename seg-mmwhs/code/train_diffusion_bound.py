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
from torch.utils.data import DataLoader
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


def get_current_mu(epoch):
    if args.mu_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.rampup_epoch is None:
            args.rampup_epoch = args.max_epoch
        return args.mu * sigmoid_rampup(epoch, args.rampup_epoch)
    else:
        return args.mu


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


def make_loader(split, dst_cls=DatasetAllTasks, repeat=None, is_training=True, unlabeled=False, task="", transforms_tr=None):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            transform=transforms_tr,
            task=task,
            num_cls=config.num_cls
        )
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


def make_model_all():
    # model = DiffVNet(
    #     n_channels=config.num_channels,
    #     n_classes=config.num_cls,
    #     n_filters=config.n_filters,
    #     normalization='batchnorm',
    #     has_dropout=True
    # ).cuda()
    model = SegResNet(in_channels=1, out_channels=config.num_cls, init_filters=config.n_filters).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    return model, optimizer


class Difficulty:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(self.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)

        self.last_dice = cur_dice

        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)

        cur_diff = torch.pow(cur_diff, 1/5)

        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        return weights * self.num_cls


def create_model(ema=False):
        # Network definition
    model = SegResNet(in_channels=1, out_channels=config.num_cls, init_filters=config.n_filters).cuda()
        # print(args.num_classes)

    # model = DiffVNet(
    #     n_channels=config.num_channels,
    #     n_classes=config.num_cls,
    #     n_filters=config.n_filters,
    #     normalization='batchnorm',
    #     has_dropout=True
    # ).cuda()
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
        unlabeled_loader = make_loader(args.split_unlabeled, transforms_tr=transforms_train_unlabeled, task=args.task, unlabeled=True)
        labeled_loader = make_loader(args.split_labeled, transforms_tr=transforms_train_labeled, task=args.task, repeat=len(unlabeled_loader.dataset))
    else:
        labeled_loader = make_loader(args.split_labeled, transforms_tr=transforms_train_labeled, task=args.task)
        unlabeled_loader = make_loader(args.split_unlabeled, transforms_tr=transforms_train_unlabeled, task=args.task, repeat=len(labeled_loader.dataset), unlabeled=True)
    eval_loader = make_loader(args.split_eval, task=args.task, is_training=False)



    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    # model = nn.DataParallel(create_model())
    # ema_model = nn.DataParallel(create_model(ema=True))
    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )
    # model, optimizer = make_model_all()0
    # ema_model = nn.DataParallel(create_model(ema=True))
    # make loss function
    diff = Difficulty(config.num_cls, accumulate_iters=50)

    deno_loss  = make_loss_function(args.sup_loss)
    sup_loss  = make_loss_function(args.sup_loss)
    unsup_loss  = make_loss_function(args.unsup_loss)


    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    mu = get_current_mu(0)

    best_eval = 0.0
    best_epoch = 0
    # for epoch_num in range(args.max_epoch + 1):
    #     loss_list = []
    #     loss_sup_list = []
    #     loss_diff_list = []
    #     loss_unsup_list = []

    #     model.train()
    #     for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):

    #         for D_theta_name, D_theta_params in model.decoder_theta.named_parameters():
    #             if D_theta_name in model.denoise_model.decoder.state_dict().keys():
    #                 D_xi_params = model.denoise_model.decoder.state_dict()[D_theta_name]
    #                 D_psi_params = model.decoder_psi.state_dict()[D_theta_name]
    #                 if D_theta_params.shape == D_xi_params.shape:
    #                     D_theta_params.data = args.ema_w * D_theta_params.data + (1 - args.ema_w) * (D_xi_params.data + D_psi_params.data) / 2.0


    #         optimizer.zero_grad()
    #         image_l, label_l = fetch_data(batch_l)
    #         label_l = label_l.long()
    #         image_u = fetch_data(batch_u, labeled=False)

    #         if args.mixed_precision:
    #             with autocast():
    #                 shp = (config.batch_size, config.num_cls)+config.patch_size

    #                 label_l_onehot = torch.zeros(shp).cuda()
    #                 label_l_onehot.scatter_(1, label_l, 1)
    #                 x_start = label_l_onehot * 2 - 1
    #                 x_t, t, noise = model(x=x_start, pred_type="q_sample")

    #                 p_l_xi = model(x=x_t, step=t, image=image_l, pred_type="D_xi_l")
    #                 p_l_psi = model(image=image_l, pred_type="D_psi_l")

    #                 L_deno = deno_loss(p_l_xi, label_l)


    #                 weight_diff = diff.cal_weights(p_l_xi.detach(), label_l)
    #                 sup_loss.update_weight(weight_diff)
    #                 L_diff = sup_loss(p_l_psi, label_l)

    #                 with torch.no_grad():
    #                     p_u_xi = model(image_u, pred_type="ddim_sample")
    #                     p_u_psi = model(image_u, pred_type="D_psi_l")
    #                     smoothing = GaussianSmoothing(config.num_cls, 3, 1)
    #                     p_u_xi = smoothing(F.gumbel_softmax(p_u_xi, dim=1))
    #                     p_u_psi = F.softmax(p_u_psi, dim=1)
    #                     pseudo_label = torch.argmax(p_u_xi + p_u_psi, dim=1, keepdim=True)


    #                 p_u_theta = model(image=image_u, pred_type="D_theta_u")
    #                 L_u = unsup_loss(p_u_theta, pseudo_label.detach())

    #                 loss = L_deno + L_diff + mu * L_u

    #             # backward passes should not be under autocast.
    #             amp_grad_scaler.scale(loss).backward()
    #             amp_grad_scaler.step(optimizer)
    #             amp_grad_scaler.update()


    #         else:
    #             raise NotImplementedError

    #         loss_list.append(loss.item())
    #         loss_sup_list.append(L_deno.item())
    #         loss_diff_list.append(L_diff.item())
    #         loss_unsup_list.append(L_u.item())
    max_epoch= args.max_epoch
    iterator = tqdm(range(max_epoch), ncols=70)
    batch_size_half = int(config.batch_size / 2)
    dice_loss = DiceLoss(config.num_cls)
    mse_loss = nn.MSELoss()
    max_iterations = max_epoch * len(unlabeled_loader)
    base_lr = args.base_lr
    iter_num = 0
    # for epoch_num in iterator:
    #     for i_batch, sampled_batch in enumerate(zip(labeled_loader, unlabeled_loader)):
    #         # T1 = time.time()
            #
    for epoch_num in iterator:
        for sampled_batch in labeled_loader:
            sampled_image, sampled_masks = fetch_data(sampled_batch)
            sampled_masks = sampled_masks.long()
            # sampled_image = sampled_batch["image"].cuda()
            # sampled_masks = sampled_batch["mask"].cuda()
            output = model(sampled_image)["seg"]
            loss_func = DC_and_CE_loss(w_ce = None)
            loss = loss_func(output,sampled_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - (iter_num+1) / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            print(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))
            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))
            # sampled_batch_s, sampled_batch_t = sampled_batch[0], sampled_batch[1]

            # volume_batch, label_batch_s = torch.cat((sampled_batch_s['image'],sampled_batch_t['image'])), \
            #                               torch.tensor(sampled_batch_s['label'])
            # # print(volume_batch.dtype,label_batch_s.dtype)
            # # print(volume_batch.shape)
            # # print(label_batch_s.shape)
            # volume_batch, label_batch_s = volume_batch.cuda(), label_batch_s.unsqueeze(dim = 1).cuda()
            # # print(batch_size_half)
            # volume_batch_t = volume_batch[batch_size_half:]
            # volume_batch_s = volume_batch[:batch_size_half]

            # # Supervised loss
            # # print(volume_batch_s.shape)
            # outputs_soft_s = torch.softmax(model(volume_batch_s)['seg'], dim=1)
            # # print(0,np.shape(outputs_soft_s.cpu()),np.shape(label_batch_s.cpu()))

            # src_in_trg = FDA_source_to_target(src_img=volume_batch_s,trg_img=volume_batch_t,L=0.1)
            # outputs_soft_sft = torch.softmax(model(src_in_trg)['seg'], dim=1)
            # # print("outputs_soft_s",outputs_soft_s.shape)
            # # print("label_batch_s",label_batch_s.shape)
            # loss_sup = dice_loss(outputs_soft_s, label_batch_s)
            # loss_supft = dice_loss(outputs_soft_sft, label_batch_s)

            # # Inter-domain
            # trg_in_src = FDA_source_to_target(src_img=volume_batch_t,trg_img=volume_batch_s,L=0.1)
            # mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5), random_aspect_ratio=False,
            #                                   n_boxes=1, invert=True)
            # cut_mask = mask_generator.generate_params(int(batch_size_half / 2),(volume_batch_t.shape[-3],
            #                                            volume_batch_t.shape[-2],
            #                                            volume_batch_t.shape[-1]))
            # mask = torch.tensor(cut_mask).type(torch.FloatTensor).cuda()
            # volume_batch_t0 = volume_batch_t[0:batch_size_half // 2, ...]
            # volume_batch_t1 = volume_batch_t[batch_size_half // 2:, ...]
            # batch_tx_mixed = volume_batch_t0 * (1.0 - mask) + \
            #                  volume_batch_t1 * mask
            # # outputs_soft_tx = torch.unsqueeze(torch.softmax(model(batch_tx_mixed)['seg'], dim=1),0)
            # outputs_soft_tx = torch.softmax(model(batch_tx_mixed)['seg'], dim=1)
            # # print("trg_in_src_shape:",np.shape(trg_in_src))
            # batch_tfs_mixed = trg_in_src[:1] * (1.0 - mask) + \
            #                   trg_in_src[1:] * mask
            # # outputs_soft_tfs = torch.unsqueeze(torch.softmax(model(batch_tfs_mixed)['seg'], dim=1),0)
            # # print(np.shape(model(batch_tfs_mixed)['seg']))
            # outputs_soft_tfs = torch.softmax(model(batch_tfs_mixed)['seg'], dim=1)
            # # Domain consistency loss


            # with torch.no_grad():
            #     ema_output = torch.softmax(ema_model(volume_batch_t)['seg'], dim=1)
            #     ema_outputfs = torch.softmax(ema_model(trg_in_src)['seg'], dim=1)
            #     ema_output_t0, ema_output_t1 = ema_output[:batch_size_half // 2],\
            #                                    ema_output[batch_size_half // 2:]
            #     batch_pred_mixed = ema_output_t0 * (1.0 - mask) + \
            #                        ema_output_t1 * mask
            #     ema_output_tfs0, ema_output_tfs1 = ema_outputfs[:batch_size_half // 2],\
            #                                        ema_outputfs[batch_size_half // 2:]
            #     batch_pred_mixed2 = ema_output_tfs0 * (1.0 - mask) + \
            #                         ema_output_tfs1 * mask
            # # print(np.shape(outputs_soft_tfs),np.unique(outputs_soft_tfs.cpu().detach().numpy()))
            # # print(np.shape(outputs_soft_tx),np.unique(outputs_soft_tx.cpu().detach().numpy()))
            # # print(np.shape(batch_pred_mixed),np.unique(batch_pred_mixed.cpu().detach().numpy()))
            # # print(np.shape(batch_pred_mixed2),np.unique(batch_pred_mixed2.cpu().detach().numpy()))
            # # print(outputs_soft_tx, outputs_soft_tfs)
            # # print(batch_pred_mixed)
            # # print(batch_pred_mixed2)
            # # print(mse_loss(outputs_soft_tfs, batch_pred_mixed))
            # # print(mse_loss(outputs_soft_tx, batch_pred_mixed2))
            # inter_loss = mse_loss(outputs_soft_tfs, batch_pred_mixed) + \
            #              mse_loss(outputs_soft_tx, batch_pred_mixed2)
            # consistency_weight = get_current_consistency_weight(epoch_num)

            # Total loss
            # print("loss_sup:",loss_sup)
            # print("loss_supft:",loss_supft)
            # print("consistency_weight",consistency_weight)
            # print("inter_loss:",inter_loss)
            # print("loss:",loss)
            # loss = loss_sup + loss_supft + consistency_weight * inter_loss

            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # lr_ = base_lr * (1.0 - (iter_num+1) / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            # iter_num = iter_num + 1
            # print(
            #     'iteration %d : loss : %f' %
            #     (iter_num, loss.item()))
            # logging.info(
            #     'iteration %d : loss : %f' %
            #     (iter_num, loss.item()))
            loss_list = []
            loss_sup_list= []
            loss_supft_list = []
            inter_loss_list = []
            loss_list.append(loss.item())
            # loss_sup_list.append(loss_sup.item())
            # loss_supft_list.append(loss_supft.item())
            # inter_loss_list.append(inter_loss.item())

        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/deno', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/unsup', np.mean(inter_loss_list), epoch_num)
        writer.add_scalar('loss/diff', np.mean(loss_supft_list), epoch_num)
        # writer.add_scalars('class_weights', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_diff))), epoch_num)

        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)} | lr : {get_lr(optimizer)} | mu : {mu}')
        # logging.info(f"     diff_w: {print_func(weight_diff)}")
        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        mu = get_current_mu(epoch_num)


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
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break


                #     volume_batch, label,spacing = \
                #         sampled_batch['image'], sampled_batch['mask'],sampled_batch['spacing']
                #     # print(spacing)
                #     spacing = [float(item) for item in spacing]
                #     label = label.squeeze(1).squeeze(0).cpu().detach().numpy()       
                #     with torch.no_grad():
                #         out = model(volume_batch.cuda())['seg']
                #         # print(0,np.unique(out.cpu().detach().numpy()),np.shape(out.cpu().detach().numpy()))
                #         out = torch.softmax(out, dim=1)
                #         # print(0,np.unique(out.cpu().detach().numpy()),np.shape(out.cpu().detach().numpy()))
                #         out = torch.argmax(out, dim=1).squeeze(0)
                #         out = out.cpu().detach().numpy()
                #         # print(np.unique(out))
                #     # print(out.shape,label.shape)
                #     print(np.unique(out),np.unique(label))
                #     metric_i = metric_()
                #     surface_dice = Metric_(spacing)
                #     with open(test_save_path + '/results.txt', 'a') as f:
                #         f.write('test_case: ' + str(i_batch) + '\n')
                #         f.write('mean_dice: ' + str(metric_i.mean().item()) + '\n')
                #         f.write('mean normalized_surface_dice: ' + str(surface_dice.mean().item()) + '\n')
                #     print('test_case %d : mean_dice : %f' % (i_batch, metric_i.mean()))
                #     print('test_case %d : normalized_dice : %f' % (i_batch, surface_dice.mean()))
                #     metric_list += metric_i
                #     dice_all+=surface_dice
                #     # metric_list1 += metric_i
                #     # dice1+=surface_dice
                #     # else:
                #     #     dice2+=surface_dice
                #     #     metric_list2 += metric_i

                # metric_list = metric_list / len(valid_loader)
                # dice_all=dice_all / len(valid_loader)
                # # metric_list1 = metric_list1 / 15
                # # metric_list2 = metric_list2 / 25
                # # dice1=dice1/15
                # # dice2=dice2/25
                # performance = metric_list.mean()
                # # performance1 = metric_list1.mean()
                # # performance2 = metric_list2.mean()
                # Performance=dice_all.mean()
                # # Performance1=dice1.mean()
                # # Performance2=dice2.mean()
                # with open(test_save_path + '/results.txt', 'a') as f:
                #     num = 1
                #     for file1,file2 in zip(metric_list,dice_all):
                #         f.write('class: ' + str(num) + '\n')
                #         f.write('dice: ' + str(file1.item()) + '\n')
                #         f.write('normalized surface dice: ' + str(file2.item()) + '\n')
                #         num += 1
                #     f.write('Total mean_dice : ' + str(performance.item())+'+-'+str(metric_list.std().item()) + '\n')
                #     # f.write('Total mean_dice normal : ' + str(performance1.item())+'+-'+str(metric_list1.std().item()) + '\n')
                #     # f.write('Total mean_dice abnormal : ' + str(performance2.item())+'+-'+str(metric_list2.std().item())+ '\n')
                #     f.write('Total normalized_surface_dice: ' + str(Performance.item())+'+-'+str(dice_all.std().item()) + '\n')
                #     # f.write('Total normalized_surface_dice normal : ' + str(Performance1.item()) +'+-'+str(dice1.std().item())+ '\n')
                #     # f.write('Total normalized_surface_dice abnormal: ' + str(Performance2.item())+'+-'+str(dice2.std().item())+'\n')
                # return round(performance.item(), 4),round(Performance.item(), 4)
    writer.close()
