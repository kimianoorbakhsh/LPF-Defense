"""Targeted point adding attack."""

import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import importlib
import sys
#sys.path.append('../')

from config import BEST_WEIGHTS
from config import MAX_ADD_BATCH as BATCH_SIZE
from dataset import ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import str2bool, set_seed
from attack import CWAdd
from attack import CrossEntropyAdvLoss, LogitsAdvLoss
from attack import ChamferDist, HausdorffDist
from DGCNN_cls import DGCNN
from pointconv import PointConvDensityClsSsg as PointConvClsSsg

def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    num = 0
    for pc, label, target in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)
            target_label = target.long().cuda(non_blocking=True)

        # attack!
        _, best_pc, success_num = attacker.attack(pc, target_label)

        # results
        num += success_num
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())
        all_target_lbl.append(target_label.detach().cpu().numpy())

    # accumulate results
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(all_target_lbl, axis=0)  # [num_data]
    return all_adv_pc, all_real_lbl, all_target_lbl, num


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='data/attack_data.npz')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40'])
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=0.,
                        help='min margin in logits adv loss')
    parser.add_argument('--dist_func', type=str, default='chamfer',
                        choices=['chamfer', 'hausdorff'],
                        help='Distance loss function to use')
    parser.add_argument('--num_add', type=int, default=512, metavar='N',
                        help='Number of points added')
    parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step', type=int, default=10, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=500, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--num_class', default=-40, type=int,
                        help='number of classes in the dataset.')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_points]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_points]
    # if args.batch_size == -1:
    #     args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    cudnn.benchmark = True
    num_class = args.num_class
    # build model
    if args.model.lower() == 'dgcnn':
        k = 20
        emb_dims = 1024
        dropout_p = 0.5
        model = DGCNN(k, emb_dims, dropout_p, output_channels=40).to('cuda:0')
        checkpoint = torch.load('pretrain/DGCNN_checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model.lower() == 'pointnet':
        model_name = 'pointnet_cls'
        cls = importlib.import_module(model_name)

        model = cls.get_model(num_class, normal_channel=False)

        # checkpoint = torch.load('pretrain/pointnet_best_model.pth')
        checkpoint = torch.load('pretrain/pointnet_best_model_shape.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model.lower() == 'pointnet2':
        model_name = 'pointnet2_cls_ssg'
        cls = importlib.import_module(model_name)

        model = cls.get_model(num_class, normal_channel=False)

        checkpoint = torch.load('pretrain/pointnet2_best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model.lower() == 'pointconv':
        model = PointConvClsSsg(num_class).cuda()
        checkpoint = torch.load('pretrain/pointconv_modelnet40-0.922204-0077.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Model not recognized')
        exit(-1)

    
    # distributed mode on multiple GPUs!
    # much faster than nn.DataParallel
    model = DistributedDataParallel(
        model.cuda(), device_ids=[args.local_rank])

    # setup attack settings
    if args.adv_func == 'logits':
        adv_func = LogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    # hyper-parameters from their official tensorflow code
    if args.dist_func == 'chamfer':
        dist_func = ChamferDist(method='adv2ori')
        init_w = 5e3
        upper_w = 4e4
    else:
        dist_func = HausdorffDist(method='adv2ori')
        init_w = 2e2
        upper_w = 9e2
    attacker = CWAdd(model, adv_func, dist_func,
                     attack_lr=args.attack_lr,
                     init_weight=init_w, max_weight=upper_w,
                     binary_step=args.binary_step,
                     num_iter=args.num_iter,
                     num_add=args.num_add)

    # attack
    test_set = ModelNet40Attack(args.data_root, num_points=args.num_points,
                                normalize=True)
    test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False,
                             sampler=test_sampler)

    # run attack
    attacked_data, real_label, target_label, success_num = attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    # save results
    save_path = './attack/results/{}_{}/Add/{}'.\
        format(args.dataset, args.num_points, args.dist_func)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.adv_func == 'logits':
        args.adv_func = 'logits_kappa={}'.format(args.kappa)
    save_name = 'Add-{}-{}-success_{:.4f}-rank_{}.npz'.\
        format(args.model, args.adv_func,
               success_rate, args.local_rank)
    print(save_name)
    np.savez(os.path.join(save_path, save_name),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8),
             target_label=target_label.astype(np.uint8))
