import argparse
import os.path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.metrics import SurfaceDiceMetric as NSD
from torch.utils.data import DataLoader

from dataloader_seg import BaseDataSets
from resunet import ResUnet

def custom_ce_loss(input, target, n_classes=4):
    log_probs = F.log_softmax(input, dim=1)
    loss = 0

    for c in range(n_classes):
        class_mask = (target == c).float()
        class_log_probs = log_probs[:, c, :, :, :]
        class_loss = -torch.sum(class_log_probs * class_mask)
        num_voxels = torch.sum(class_mask)
        if num_voxels > 0:
            class_loss /= num_voxels
        loss += class_loss

    return loss / n_classes

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0.0)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import surface_distance as surfdist

def one_hot_logits(logits):
    # 获取每个位置的最大logits值对应的类别索引
    _, predicted = torch.max(logits, dim=1, keepdim=True)

    # 创建one-hot编码
    one_hot = torch.zeros_like(logits, dtype=torch.float32)
    return one_hot.scatter_(1, predicted, 1)

def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.stack(tensor_list, dim=1)
    return output_tensor.float()

def _show_nsd(input, target, spacing_mm=None):
    input=input.unsqueeze(0)
    target = target.unsqueeze(0)
    input=one_hot_encoder(input,n_classes=8)
    input = one_hot_logits(input)
    target = one_hot_encoder(target, n_classes=8)
    compute_nsd = NSD(class_thresholds=[1 for _ in range(7)], reduction='mean_channel')
    compute_nsd(input, target, spacing=spacing_mm)
    nsd = compute_nsd.aggregate(reduction="none")
    #print(nsd)
    nsd=nsd.squeeze()
    return nsd

def evaluate_model(model, model_name, valid_loader, eval_flag='l', test_save_path=None):
    """
    eval flag: l: landmark, s: segmentation, mt: both l and s
    """

    distance_losses = [[] for i in range(6)]
    model.eval()
    dice_list = []
    data_idx = 0
    if eval_flag == 's':
        def metric_():
            import torchmetrics
            metric_list = torchmetrics.functional.dice(torch.from_numpy(out).cuda(),
                                                       torch.from_numpy(label).cuda(),
                                                       average='none', num_classes=8)
            #print("torch_label:",torch.from_numpy(label).shape,"torch_out:",torch.from_numpy(out).shape)
            return metric_list[1:]
        def Metric_():
            surface_dice = _show_nsd(torch.from_numpy(out).cuda(),torch.from_numpy(label).cuda(),(0.5,0.5,0.5))
            return surface_dice
        metric_list = 0.0
        metric_list1 = 0.0
        metric_list2 = 0.0
        dice_all = 0.0
        dice1 = 0.0
        dice2 = 0.0
        for i_batch, sampled_batch in enumerate(valid_loader):
            volume_batch, label = \
                sampled_batch['image'], sampled_batch['mask']
            label = label.squeeze(1).squeeze(0).cpu().detach().numpy()
            with torch.no_grad():
                out = model(volume_batch.cuda())
                out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                # if True:
                #     bbox = sampled_batch['bbox']
                #     out = ndimage.zoom(out, zoom=
                #     ((int(bbox[0][1]) - int(bbox[0][0])) / (out.shape[0]),
                #      (int(bbox[1][1]) - int(bbox[1][0])) / (out.shape[1]),
                #      (int(bbox[2][1]) - int(bbox[2][0])) / (out.shape[2])), order=0)
                #     out = np.pad(out, ((int(bbox[0][0]), label.shape[0] - int(bbox[0][1])),
                #                        (int(bbox[1][0]), label.shape[1] - int(bbox[1][1])),
                #                        (int(bbox[2][0]), label.shape[2] - int(bbox[2][1]))))

            metric_i = metric_()
            surface_dice = Metric_()
            with open(test_save_path + '/results.txt', 'a') as f:
                f.write('test_case: ' + str(i_batch) + '\n')
                f.write('mean_dice: ' + str(metric_i.mean().item()) + '\n')
                f.write('mean normalized_surface_dice: ' + str(surface_dice.mean().item()) + '\n')
            print('test_case %d : mean_dice : %f' % (i_batch, metric_i.mean()))
            print('test_case %d : normalized_dice : %f' % (i_batch, surface_dice.mean()))
            metric_list += metric_i
            dice_all+=surface_dice
            if i_batch < 15:
                metric_list1 += metric_i
                dice1+=surface_dice
            else:
                dice2+=surface_dice
                metric_list2 += metric_i

        metric_list = metric_list / len(valid_loader)
        dice_all=dice_all / len(valid_loader)
        metric_list1 = metric_list1 / 15
        metric_list2 = metric_list2 / 25
        dice1=dice1/15
        dice2=dice2/25
        #print(metric_list)
        #print(dice_all)
        performance = metric_list.mean()
        performance1 = metric_list1.mean()
        performance2 = metric_list2.mean()
        Performance=dice_all.mean()
        Performance1=dice1.mean()
        Performance2=dice2.mean()
        with open(test_save_path + '/results.txt', 'a') as f:
            num = 1
            for file1,file2 in zip(metric_list,dice_all):
                f.write('class: ' + str(num) + '\n')
                f.write('dice: ' + str(file1.item()) + '\n')
                f.write('normalized surface dice: ' + str(file2.item()) + '\n')
                num += 1
            f.write('Total mean_dice : ' + str(performance.item())+'+-'+str(metric_list.std().item()) + '\n')
            f.write('Total mean_dice normal : ' + str(performance1.item())+'+-'+str(metric_list1.std().item()) + '\n')
            f.write('Total mean_dice abnormal : ' + str(performance2.item())+'+-'+str(metric_list2.std().item())+ '\n')
            f.write('Total normalized_surface_dice: ' + str(Performance.item())+'+-'+str(dice_all.std().item()) + '\n')
            f.write('Total normalized_surface_dice normal : ' + str(Performance1.item()) +'+-'+str(dice1.std().item())+ '\n')
            f.write('Total normalized_surface_dice abnormal: ' + str(Performance2.item())+'+-'+str(dice2.std().item())+'\n')
        return round(performance.item(), 4), performance1.item(),round(Performance.item(), 4)


def train(args, model_name='resunet', visualization=False):
    setup_seed(1234)

    # load dataset for semantic segmentation
    train_seg = BaseDataSets(data_dir=args.root_path_s, mode="train", list_name=args.train_data_s,
                             patch_size=args.patch_size, crop=args.crop, zoom=args.zoom, transform=None, type=args.type)
    valid_seg = BaseDataSets(data_dir=args.root_path_t, mode="test", list_name=args.train_data_t,
                             patch_size=args.patch_size, crop=args.crop, zoom=args.zoom, transform=None, type=args.type)
    test_seg = BaseDataSets(data_dir=args.root_path_t, mode="test", list_name=args.test_data,
                            patch_size=args.patch_size, crop=args.crop, zoom=args.zoom, type=args.type)

    # atlas
    train_loader_seg = DataLoader(train_seg, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    # feta valid
    valid_loader_seg = DataLoader(valid_seg, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    # feta test
    test_loader_seg = DataLoader(test_seg, batch_size=1, shuffle=False, num_workers=1)

    print('Data have been loaded.')
    model = ResUnet(num_classes=8)
    model=model.cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    confidence_threshold = 0.9
    local_best = 0

    for epoch in range(100):
        print('epoch:', epoch)
        total_train_loss = 0
        model.train()
        for i_batch, sampled_batch in enumerate(train_loader_seg):
            data_seg = sampled_batch
            voxel_seg, gt_seg = data_seg['image'].cuda(), data_seg['mask'].cuda()
            voxel_fuse = voxel_seg  # landmark before, seg after
            # print(torch.max(gt_seg))
            out_seg = model(voxel_fuse)
            loss_seg = F.cross_entropy(out_seg, gt_seg.squeeze(1).long())
            loss = loss_seg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
        print(loss)

        save_path = 'runs/' + model_name + args.type + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + '/results.txt', 'a') as f:
            f.write("epoch:"+str(epoch) + '\n')
        valid_dice, valid_abn_dice ,valid_nsd= evaluate_model(model, model_name, valid_loader_seg, eval_flag='s', test_save_path=save_path)
        if local_best < valid_dice:
            local_best = valid_dice
            test_dice, test_abn_dice ,test_nsd= evaluate_model(model, model_name, test_loader_seg, eval_flag='s', test_save_path=save_path)
            weights_path = 'runs/' + model_name + '/' + str(epoch) +'-valid-'+ str(valid_dice) + '-test-' + str(test_dice) + '.pth'
            torch.save(model.state_dict(), weights_path)
        print('Dice loss:', valid_dice)
        print('Normalized surface dice loss:', valid_nsd)

    return 'finish'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str,
                        default='', help='Dataset type')
    parser.add_argument('--root_path_t', type=str,
                        default='./seg_dataset/feta', help='Name of Experiment')
    parser.add_argument('--root_path_s', type=str,
                        default='./seg_dataset/atlas', help='Name of Experiment')
    parser.add_argument('--train_data_s', type=str,
                        default='train.list', help='Name of Dataset')
    parser.add_argument('--train_data_t', type=str,
                        default='train.list', help='Name of Dataset')
    parser.add_argument('--test_data', type=str,
                        default='test.list', help='Name of Dataset')
    parser.add_argument('--exp', type=str,
                        default='ASC', help='Name of Experiment')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='output channel of network')
    parser.add_argument('--max_epoch', type=int,
                        default=100, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size per gpu')
    parser.add_argument('--patch_size', type=list, default=[128, 128, 128],
                        help='patch size of network input')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.0001,
                        help='segmentation network learning rate')
    parser.add_argument('--epoch_gap', type=int, default=5,
                        help='choose epoch gap to val model')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
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

    # setting for TransDoDNet
    parser.add_argument("--using_transformer", type=str2bool, default=True)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=768, type=int)
    parser.add_argument('--hidden_dim', default=192, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=2, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_feature_levels', default=3, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--normalize_before', default=False, type=str2bool)
    parser.add_argument('--deepnorm', default=True, type=str2bool)
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument("--add_memory", type=int, default=2, choices=(0,1,2)) # feature fusion: 0: cnn; 1:tr; 2:cnn+tr

    parser.add_argument('--res_depth', default=50, type=int)
    parser.add_argument("--dyn_head_dep_wid", type=str, default='3,8')
    
    args = parser.parse_args()

    train(args, 'resunet_seg')
