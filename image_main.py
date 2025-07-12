import sys

import numpy as np
import argparse
import os
import torch

import math
import json

import attack_csma

# import TPAMI_attack
from datasets import get_dataset
from gluoncv.torch.model_zoo import get_model
from utils import CONFIG_PATHS, OPT_PATH, get_cfg_custom
import pickle as pkl
import torchvision


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    # parallel run
    parser.add_argument('--batch_nums', type=int, default=1)
    parser.add_argument('--batch_index', type=int, default=1)

    # parser.add_argument('--adv_path', type=str, default='', help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for reference (default: 16)')
    parser.add_argument('--attack_method', type=str, default='I2V_MF',
                        help='I2V_MF, AENS_I2V_MF')
    # parser.add_argument('--step', type=int, default=60, metavar='N',
    #                 help='Multi-step or One-step in TI and SGM.')
    parser.add_argument('--step', type=int, default=60, metavar='N',
                        help='Multi-step or One-step in TI and SGM.')
    parser.add_argument('--file_prefix', type=str, default='alexnet_Kinetics_i2vmf-adv-self-cross-patch2-0.5')

    # for std
    parser.add_argument('--depth', type=int, default=2, help='1,2,3,4')
    parser.add_argument('--lamb', type=float, default=0.1, help='')

    parser.add_argument('--mode', type=str, default='direction', help='diff_norm\direction')
    parser.add_argument('--step_size', type=float, default=0.005, help='')

    # for dropout
    parser.add_argument('--dropout', type=float, default=0.1, help='')

    # for mix mask
    parser.add_argument('--mix_factor', type=float, default=0.4, help='')
    parser.add_argument('--mix_type', type=str, default='global', help='')

    # for direction with changing image model
    parser.add_argument('--direction_image_model', type=str, default='alexnet',
                        help='resnet, densenet, squeezenet, vgg, alexnet')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, '{}-{}-{}-{}'.format('Image', args.attack_method, args.step, args.file_prefix))
    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    return args


if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)
    # loading cfg.
    cfg_path = CONFIG_PATHS['i3d_resnet101']
    cfg = get_cfg_custom(cfg_path, args.batch_size)

    # loading dataset and model.
    dataset_loader = get_dataset(cfg)
    # sys.exit()
    nums_contained = int(400 / args.batch_nums)
    left = (args.batch_index - 1) * nums_contained
    right = args.batch_index * nums_contained

    # attack ImageGuidedStd_Adam ImageGuidedFMDirection_Adam ImageGuidedFML2_Adam_MultiModels ENS_FT_I2V ILAF

    if args.attack_method == 'I2V_MF':
        model_name_lists = [args.direction_image_model]
        depths = {
            args.direction_image_model: [2, 3]
        }
        attack_method = getattr(TPAMI_attack_improve, args.attack_method)(model_name_lists, depths=depths, step_size=args.step_size, mix_factor=args.mix_factor, mix_type=args.mix_type)

    elif args.attack_method == 'AENS_I2V_MF':
        model_name_lists = ['resnet', 'vgg', 'squeezenet', 'alexnet']
        depths = {
            'resnet': [2, 3],
            'vgg': [2, 3],
            'squeezenet': [2, 3],
            'alexnet': [2, 3]
        }
        attack_method = getattr(TPAMI_attack_improve, args.attack_method)(model_name_lists, depths=depths, step_size=args.step_size, mix_factor=args.mix_factor, mix_type=args.mix_type)

    for step, data in enumerate(dataset_loader):
        if step >= left and step < right:
            if step % 1 == 0:
                print('Running {}, {}/{}'.format(args.attack_method, step + 1, len(dataset_loader)))
            val_batch = data[0]
            val_label = data[1]
            video_names = data[2]
            all_index = data[4]
            # print(all_index)
            # sys.exit()
            adv_batches = attack_method(val_batch, val_label, video_names)
            # adv_batches = val_batch
            print(adv_batches.size())

            # print(adv_batches.max(), adv_batches.min())
            # b, c, f, h, w = adv_batches.shape
            # image_inps = adv_batches.permute([0, 2, 1, 3, 4])
            # image_inps = image_inps.reshape(b * f, c, h, w)
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            # mean = torch.as_tensor(mean, dtype=image_inps.dtype).cuda()
            # std = torch.as_tensor(std, dtype=image_inps.dtype).cuda()
            # image_inps.mul_(std[:, None, None]).add_(mean[:, None, None])
            # print(image_inps.max(), image_inps.min())
            # torchvision.utils.save_image(image_inps, os.path.join(args.adv_path, '{}-vis.png'.format(val_label[0].item())))
            # sys.exit()

            for ind, label in enumerate(val_label):
                adv = adv_batches[ind].detach().cpu().numpy()
                np.save(os.path.join(args.adv_path, '{}-adv'.format(label.item())), adv)
            # for ind, label in enumerate(val_label):
            #     np.save(os.path.join(args.adv_path, '{}-adv'.format(label.item())), all_index)

    with open(os.path.join(args.adv_path, 'loss_info_{}.json'.format(args.batch_index)), 'w') as opt:
        json.dump(attack_method.loss_info, opt)
