import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm
from utils import read_list, read_nifti
from utils import config
import numpy as np
from scipy.ndimage import label as label_mask

def retain_largest_connected_component(pred, threshold=0.5):
    binary_pred = (pred > threshold).astype(int)
    labeled_array, num_features = label_mask(binary_pred)
    if num_features == 0:
        return binary_pred
    sizes = np.bincount(labeled_array.ravel())
    largest_component_label = sizes[1:].argmax() + 1
    largest_component = (labeled_array == largest_component_label)
    return largest_component.astype(pred.dtype)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='synapse')
    parser.add_argument('--exp', type=str, default="fully")
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--modality', type=str, default='CT')

    args = parser.parse_args()

    config = config.Config(args.task)

    ids_list = read_list(args.split, task=args.task)
    results_all_folds = []

    txt_path = "./logs/"+args.exp+"/evaluation_res.txt"
    print("\n Evaluating...")
    fw = open(txt_path, 'w')
    for fold in range(1, args.folds+1):

        test_cls = [i for i in range(1, config.num_cls)]
        values = np.zeros((len(ids_list), len(test_cls), 2)) # dice and asd

        for idx, data_id in enumerate(tqdm(ids_list)):
            pred = read_nifti(os.path.join("./logs",args.exp, "fold"+str(fold), "predictions",f'{data_id}.nii.gz'))
            pred = retain_largest_connected_component(pred) * pred
            lb_path = os.path.join(config.save_dir, 'processed', f'{data_id}_label.nii.gz')
            label = read_nifti(lb_path)

            # label = np.load(lb_path)

            padding_flag = label.shape[0] < config.patch_size[0] or label.shape[1] < config.patch_size[1] or label.shape[2] < config.patch_size[2]
            # pad the sample if necessary
            if padding_flag:
                pw = max((config.patch_size[0] - label.shape[0]) // 2 + 1, 0)
                ph = max((config.patch_size[1] - label.shape[1]) // 2 + 1, 0)
                pd = max((config.patch_size[2] - label.shape[2]) // 2 + 1, 0)
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            dd, ww, hh = label.shape

            for i in test_cls:
                pred_i = (pred == i)
                label_i = (label == i)
                if pred_i.sum() > 0 and label_i.sum() > 0:
                    dice = metric.binary.dc(pred == i, label == i) * 100
                    hd95 = metric.binary.asd(pred == i, label == i)
                    values[idx][i-1] = np.array([dice, hd95])
                elif pred_i.sum() > 0 and label_i.sum() == 0:
                    dice, hd95 = 0, 128
                elif pred_i.sum() == 0 and label_i.sum() > 0:
                    dice, hd95 =  0, 128
                elif pred_i.sum() == 0 and label_i.sum() == 0:
                    dice, hd95 =  1, 0

                values[idx][i-1] = np.array([dice, hd95])
        values_mean_cases = np.mean(values, axis=0)
        results_all_folds.append(values)
        fw.write("Fold" + str(fold) + '\n')
        fw.write("------ Dice ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,0],1)) + '\n')
        fw.write("------ ASD ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,1],1)) + '\n')
        fw.write('Average Dice:'+str(np.mean(values_mean_cases, axis=0)[0]) + '\n')
        fw.write('Average  ASD:'+str(np.mean(values_mean_cases, axis=0)[1]) + '\n')
        fw.write("=================================")
        print("Fold", fold)
        print("------ Dice ------")
        print(np.round(values_mean_cases[:,0],1))
        print("------ ASD ------")
        print(np.round(values_mean_cases[:,1],1))
        print(np.mean(values_mean_cases, axis=0)[0], np.mean(values_mean_cases, axis=0)[1])

    results_all_folds = np.array(results_all_folds)

    fw.write('\n\n\n')
    fw.write('All folds' + '\n')

    results_folds_mean = results_all_folds.mean(0)

    for i in range(results_folds_mean.shape[0]):
        fw.write("="*5 + " Case-" + str(ids_list[i]) + '\n')
        fw.write('\tDice:'+str(np.round(results_folds_mean[i][:,0],2).tolist()) + '\n')
        fw.write('\t ASD:'+str(np.round(results_folds_mean[i][:,1],2).tolist()) + '\n')
        fw.write('\t'+'Average Dice:'+str(np.mean(results_folds_mean[i], axis=0)[0]) + '\n')
        fw.write('\t'+'Average  ASD:'+str(np.mean(results_folds_mean[i], axis=0)[1]) + '\n')

    fw.write("=================================\n")
    fw.write('Final Dice of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,0].tolist()]) + '\n')
    fw.write('Final ASD of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,1].tolist()]) + '\n')
    print("=================================")
    print('Final Dice of each class')
    print(str([round(x,1) for x in results_folds_mean.mean(0)[:,0].tolist()]))
    print('Final ASD of each class')
    print(str([round(x,1) for x in results_folds_mean.mean(0)[:,1].tolist()]))
    std_dice = np.std(results_all_folds.mean(1).mean(1)[:,0])
    std_hd = np.std(results_all_folds.mean(1).mean(1)[:,1])

    fw.write('Final Avg Dice: '+str(round(results_folds_mean.mean(0).mean(0)[0], 2)) +'±' +  str(round(std_dice,2)) + '\n')
    fw.write('Final Avg  ASD: '+str(round(results_folds_mean.mean(0).mean(0)[1], 2)) +'±' +  str(round(std_hd,2)) + '\n')

    print('Final Avg Dice: '+str(round(results_folds_mean.mean(0).mean(0)[0], 2)) +'±' +  str(round(std_dice,2)))
    print('Final Avg  ASD: '+str(round(results_folds_mean.mean(0).mean(0)[1], 2)) +'±' +  str(round(std_hd,2)))

# 计算每个类别的Dice和ASD的均值和标准差
    class_metrics = np.zeros((config.num_cls - 1, 2, 2))  # 类别, 评价指标(Dice, ASD), 统计数据(均值, 标准差)

    for cls in range(1, config.num_cls):
        cls_dice = results_all_folds[:, :, cls-1, 0]  # 所有折叠和样本中当前类别的Dice值
        cls_asd = results_all_folds[:, :, cls-1, 1]  # 所有折叠和样本中当前类别的ASD值

        dice_mean = np.mean(cls_dice)
        dice_std = np.std(cls_dice)
        asd_mean = np.mean(cls_asd)
        asd_std = np.std(cls_asd)

        class_metrics[cls-1, 0, 0] = dice_mean  # Dice均值
        class_metrics[cls-1, 0, 1] = dice_std   # Dice标准差
        class_metrics[cls-1, 1, 0] = asd_mean   # ASD均值
        class_metrics[cls-1, 1, 1] = asd_std    # ASD标准差

        # 将结果写入文件
        fw.write(f"\nClass {cls} Metrics:\n")
        fw.write(f"Dice - Mean: {dice_mean:.2f}, Std Dev: {dice_std:.2f}\n")
        fw.write(f"ASD - Mean: {asd_mean:.2f}, Std Dev: {asd_std:.2f}\n")

    fw.close()  # 确保文件写入完成后关闭文件

    # 打印到控制台，供立即查看
    print("\nDetailed Dice and ASD metrics by class:")
    for cls in range(1, config.num_cls):
        print(f"Class {cls}:")
        print(f"  Dice - Mean: {class_metrics[cls-1, 0, 0]:.2f}, Std Dev: {class_metrics[cls-1, 0, 1]:.2f}")
        print(f"  ASD - Mean: {class_metrics[cls-1, 1, 0]:.2f}, Std Dev: {class_metrics[cls-1, 1, 1]:.2f}")

