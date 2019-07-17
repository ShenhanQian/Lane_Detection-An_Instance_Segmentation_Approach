"""
处理tusimple数据集脚本
"""
import argparse
import glob
import json
import os
import os.path as ops
import shutil

import cv2
import numpy as np


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='The origin path of unzipped tusimple dataset')
    parser.add_argument('--tag', type=str, help='Explanation of dataset details')

    return parser.parse_args()


def process_json_file(json_file_path, src_dir, ori_dst_dir, binary_dst_dir, instance_dst_dir):
    """

    :param json_file_path:
    :param src_dir: 原始clips文件路径
    :param ori_dst_dir: rgb训练样本
    :param binary_dst_dir: binary训练标签
    :param instance_dst_dir: instance训练标签
    :return:
    """
    assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

    image_nums = len(os.listdir(ori_dst_dir))
    # thickness = 16
    thickness = 5

    with open(json_file_path, 'r') as file:
        for line_index, line in enumerate(file):
            info_dict = json.loads(line)

            image_dir = ops.split(info_dict['raw_file'])[0]
            image_dir_split = image_dir.split('/')[1:]
            image_dir_split.append(ops.split(info_dict['raw_file'])[1])
            image_name = '_'.join(image_dir_split)
            image_path = ops.join(src_dir, info_dict['raw_file'])
            assert ops.exists(image_path), '{:s} not exist'.format(image_path)

            h_samples = info_dict['h_samples']
            lanes = info_dict['lanes']

            image_name_new = '{:s}.png'.format('{:d}'.format(line_index + image_nums).zfill(4))

            src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            dst_binary_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
            dst_instance_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)

            for lane_index, lane in enumerate(lanes):
                assert len(h_samples) == len(lane)
                lane_x = []
                lane_y = []
                for index in range(len(lane)):
                    if lane[index] == -2:
                        continue
                    else:
                        ptx = lane[index]
                        pty = h_samples[index]
                        lane_x.append(ptx)
                        lane_y.append(pty)
                if not lane_x:
                    continue
                lane_pts = np.vstack((lane_x, lane_y)).transpose()
                lane_pts = np.array([lane_pts], np.int64)
                cv2.polylines(dst_binary_image, lane_pts, isClosed=False,
                              color=255, thickness=thickness)
                cv2.polylines(dst_instance_image, lane_pts, isClosed=False,
                              color=lane_index * 50 + 20, thickness=thickness)
            
            dst_binary_image_path = ops.join(binary_dst_dir, image_name_new)
            dst_instance_image_path = ops.join(instance_dst_dir, image_name_new)
            dst_rgb_image_path = ops.join(ori_dst_dir, image_name_new)

            # in order to speed up training process, image can be resized in advance
            dst_binary_image = cv2.resize(dst_binary_image, (512, 288), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(dst_binary_image_path, dst_binary_image)

            dst_instance_image = cv2.resize(dst_instance_image, (512, 288), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(dst_instance_image_path, dst_instance_image)

            src_image = cv2.resize(src_image, (512, 288), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(dst_rgb_image_path, src_image)

            print('Process {:s} success'.format(image_name))


def gen_train_sample(tag, src_dir, image_dir, gt_binary_dir, gt_instance_dir):
    """
    生成图像训练列表
    :param src_dir: dataset directory
    :param image_dir: input RGB image directory
    :param gt_binary_dir: binary segmentation ground-truth image directory
    :param gt_instance_dir: instance segmentation ground-truth image directory
    :return:
    """

    img_list = os.listdir(gt_binary_dir)
    num = len(img_list)
    perm = np.random.permutation(num)

    info_list = []
    for index, file_id in enumerate(perm):
        image_name = img_list[file_id]

        if not image_name.endswith('.png'):
            continue

        binary_gt_image_path = ops.join(gt_binary_dir, image_name)
        instance_gt_image_path = ops.join(gt_instance_dir, image_name)
        image_path = ops.join(image_dir, image_name)

        assert ops.exists(image_path), '{:s} not exist'.format(image_path)
        assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

        print(f'Checking {index}')

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_GRAYSCALE)
        i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_GRAYSCALE)

        '''LaneNet'''
        if b_gt_image is None or image is None or i_gt_image is None:
            print('图像对: {:s}损坏'.format(image_name))
            continue
        else:
            info = '{:s} {:s} {:s}\n'.format(image_path, binary_gt_image_path, instance_gt_image_path)
            info_list.append(info)

    with open('{:s}/training_{:s}/train.txt'.format(src_dir, tag), 'w') as file:
        for index, line in enumerate(info_list[:int(num * 0.75)]):
            file.write(line)

    with open('{:s}/training_{:s}/valid.txt'.format(src_dir, tag), 'w') as file:
        for index, line in enumerate(info_list[int(num * 0.75):]):
            file.write(line)

    return


def process_tusimple_dataset(src_dir, tag):
    """

    :param src_dir:
    :return:
    """
    print('src_dir:', src_dir)
    print('tag:', tag)

    traing_folder_path = ops.join(src_dir, 'training_%s' % tag)

    testing_folder_path = ops.join(src_dir, 'testing_%s' % tag)

    os.makedirs(traing_folder_path, exist_ok=True)
    os.makedirs(testing_folder_path, exist_ok=True)

    for json_label_path in glob.glob('{:s}/label*.json'.format(src_dir)):
        json_label_name = ops.split(json_label_path)[1]
        shutil.copyfile(json_label_path, ops.join(traing_folder_path, json_label_name))

    for json_label_path in glob.glob('{:s}/test*.json'.format(src_dir)):
        json_label_name = ops.split(json_label_path)[1]
        shutil.copyfile(json_label_path, ops.join(testing_folder_path, json_label_name))

    gt_image_dir = ops.join(traing_folder_path, 'gt_image')
    gt_binary_dir = ops.join(traing_folder_path, 'gt_binary_image')
    gt_instance_dir = ops.join(traing_folder_path, 'gt_instance_image')
    
    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)
	
    for json_label_path in glob.glob('{:s}/*.json'.format(traing_folder_path)):
        process_json_file(json_label_path, src_dir, gt_image_dir, gt_binary_dir, gt_instance_dir)

    gen_train_sample(tag, src_dir, gt_image_dir, gt_binary_dir, gt_instance_dir)

    return


if __name__ == '__main__':
    
    np.random.seed(2020)
    args = init_args()
    args.src_dir = '/root/lane_detection/dataset/tusimple/train_set'
    process_tusimple_dataset(args.src_dir, args.tag)
