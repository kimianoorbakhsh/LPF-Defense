"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data, PointSampler, Normalize, RandomNoise, RandomRotate, toTensor
from torchvision import transforms
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--test_data_path', type=str, default='Data/test/modelnet10_test_data.npy', help='test data path')
    parser.add_argument('--test_label_path', type=str, default='Data/test/modelnet10_test_labels.npy', help='test data path')
    parser.add_argument('--main_path', type=str, default='/content/drive/MyDrive/Kimia', help='main path')
    return parser.parse_args()


def test(model, testloader, device, verbose=True):
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            points, labels = data
            points = points.to(device)
            labels = labels.to(device)
            points = points.transpose(1, 2).float()

            logits = model(points)  # [B, num_classes]
            if isinstance(logits, tuple):  # PointNet
                logits = logits[0]

            # print
            pred = torch.argmax(logits, dim=-1)  # [B]
            correct += (pred == labels).sum().item()

            total += labels.shape[0]
            

        acc = 100 * (correct / total)
        return acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'Data'
    

    test_X, test_y = load_data(args.main_path, args.test_data_path, args.test_label_path, mode='test')
    test_X, test_y = test_X.astype(np.float32), test_y.astype(np.int64)
    
    default_transform = transforms.Compose([toTensor()])
    
    test_dataset = ModelNetDataLoader(test_X, test_y, transform=default_transform)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        acc = test(classifier, testDataLoader, device=torch.device('cuda'))
        log_string('Test Instance Accuracy: %f' % (acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
