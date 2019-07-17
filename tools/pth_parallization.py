import torch
import argparse
from collections import OrderedDict


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, help='path to parameter file (.pth)')

    return parser.parse_args()

def parallelize(model_static_dict):

    new_model_static_dict = OrderedDict()

    for k, v in model_static_dict.items():
        k = 'module.' + k
        new_model_static_dict[k] = v

    return new_model_static_dict


def unparallelize(model_static_dict):

    new_model_static_dict = OrderedDict()

    for k, v in model_static_dict.items():
        k = k[7:]
        new_model_static_dict[k] = v

    return new_model_static_dict


if __name__ == '__main__':

    args = init_args()

    # args.ckpt_path = r'E:\Documents\SVIP\Projects\lane-detection\code\lane-detection-pytorch\check_point/ckpt_2019-05-18_11-41-33_ENet-512-(d10-v2)-(B1-D10)-2020-noDrop/ckpt_2019-05-18_11-41-33_epoch-50.pth'

    ckpt = torch.load(args.ckpt_path)
    model_static_dict = ckpt['model_state_dict']

    if 'module' in list(model_static_dict.keys())[0]:
        new_model_static_dict = unparallelize(model_static_dict)
        new_ckpt_path = args.ckpt_path.split('.')[0] + '_unparalleled.pth'
        print("- 'module'")
        print(new_ckpt_path)
    
    else:
        new_model_static_dict = parallelize(model_static_dict)
        new_ckpt_path = args.ckpt_path.split('.')[0] + '_paralleled.pth'
        print("+ 'module'")
        print(new_ckpt_path)
    
    ckpt['model_state_dict'] = new_model_static_dict
    
    torch.save(ckpt, new_ckpt_path)

