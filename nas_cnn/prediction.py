# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import nas_cnn.utils as utils

import nas_cnn.utils
from nas_cnn.model import NetworkCIFAR as Network
import nas_cnn.genotypes as genotypes
# 工作路径
path = os.path.abspath('')
sys.path.append(path)

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str,
                    default='./img/2.jpg',
                    help='location of the data corpus')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='./img/weights.pt',
                    help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_X1', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10
data_transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
name = ["airplane", 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def main():
    start_time_all = time.time()

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    start_time = time.time()
    utils.load_cpu(model, args.model_path)
    print('load_time=', time.time() - start_time)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()

    model.drop_path_prob = args.drop_path_prob
    classes = infer(args.data, model, criterion)
    logging.info(classes)
    print('pre_time=', time.time() - start_time_all)
    return classes


def infer(data, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    # img = Image.open(data)
    # plt.imshow(img)
    # plt.show()
    # img = data_transform(img)
    im = cv2.imread(data)
    im = cv2.resize(im, dsize=(32, 32))
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    im = trans(im)
    input = im.unsqueeze(0)
    # input = torch.unsqueeze(img, dim=0)
    input = Variable(input)
    logits, _ = model(input)
    logits = logits.data.cpu().numpy()
    logits = [i for i in logits[0]]
    pre = sorted(logits)[-1]
    # print(logits)
    print('pre', pre)
    index = logits.index(pre)
    classes = name[index]

    return classes


def get_class(image_path='./img/2.jpg'):
    genotype = eval("genotypes.%s" % args.arch)
    # image_path = 'D:\github\pytorch\CIFAR10\image.png'
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    # model_path = path+'./img/weights.pt'  本地用
    model_path = path+'/weights.pt'  #服务器用
    utils.load_cpu(model, model_path)
    model.drop_path_prob = args.drop_path_prob
    criterion = nn.CrossEntropyLoss()
    classes = infer(image_path, model, criterion)
    print('----nas_cnn---')
    return classes


if __name__ == '__main__':
    #     main()
    get_class()
