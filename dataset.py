import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import cv2
import ujson as json
from model import utils


VGG_MEAN = [103.939, 116.779, 123.68]


class TuSimpleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_dir, phase, size=(512,288), transform=None):
        """
        Args:
            dataset_dir: The directory path of the dataset
            phase: 'train', 'val', or 'test'
        """
        self.dataset_dir = dataset_dir
        self.phase = phase
        self.size = size
        self.transform = transform
        assert os.path.exists(dataset_dir), 'Directory {} does not exist!'.format(dataset_dir)

        if phase == 'train' or phase == 'val':
            label_files = list()
            if phase == 'train':
                label_files.append(os.path.join(dataset_dir, 'label_data_0313.json'))
                label_files.append(os.path.join(dataset_dir, 'label_data_0531.json'))
            elif phase == 'val':
                label_files.append(os.path.join(dataset_dir, 'label_data_0601.json'))

            self.image_list = []
            self.lanes_list = []
            for file in label_files:
                try:
                    for line in open(file).readlines():
                        info_dict = json.loads(line)
                        self.image_list.append(info_dict['raw_file'])

                        h_samples = info_dict['h_samples']
                        lanes = info_dict['lanes']

                        xy_list = list()
                        for lane in lanes:
                            y = np.array([h_samples]).T
                            x = np.array([lane]).T
                            xy = np.hstack((x, y))

                            index = np.where(xy[:, 0] > 2)
                            xy_list.append(xy[index])
                        self.lanes_list.append(xy_list)
                except BaseException:
                    raise Exception(f'Fail to load {file}.')

        elif phase == 'test':
            task_file = os.path.join(dataset_dir, 'test_tasks_0627.json')
            try:
                self.image_list = [json.loads(line)['raw_file'] for line in open(task_file).readlines()]
            except BaseException:
                raise Exception(f'Fail to load {task_file}.')
        elif phase == 'test_extend':  # to load extra frames which haave no lane annotation
            task_file = os.path.join(dataset_dir, 'test_tasks_0627.json')
            try:
                self.image_list = []
                for line in open(task_file).readlines():
                    path = json.loads(line)['raw_file']
                    dir = os.path.join(dataset_dir, path[:-7])
                    for i in range(1, 21):
                        self.image_list.append(os.path.join(dir, '%d.jpg' % i))
            except BaseException:
                raise Exception(f'Fail to load {task_file}.')

        else:
            raise Exception(f"Phase '{self.phase}' cannot be recognize!")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        if self.phase == 'train' or self.phase == 'val':

            '''OpenCV'''
            img_path = os.path.join(self.dataset_dir, self.image_list[idx])
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            h, w, c = image.shape
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image -= VGG_MEAN
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float() / 255

            bin_seg_label = np.zeros((h, w), dtype=np.uint8)
            inst_seg_label = np.zeros((h, w), dtype=np.uint8)
            # inst_seg_label = np.zeros((h, w, 3), dtype=np.uint8)

            lanes = self.lanes_list[idx]
            for idx, lane in enumerate(lanes):
                cv2.polylines(bin_seg_label, [lane], False, 1, 10)
                cv2.polylines(inst_seg_label, [lane], False, idx+1, 10)  # grey, for training
                # cv2.polylines(inst_seg_label, [lane], False, utils.get_color(idx), 10)  # colored, for visualization

            bin_seg_label = cv2.resize(bin_seg_label, self.size, interpolation=cv2.INTER_NEAREST)  #
            inst_seg_label = cv2.resize(inst_seg_label, self.size, interpolation=cv2.INTER_NEAREST)

            bin_seg_label = torch.from_numpy(bin_seg_label).long()
            inst_seg_label = torch.from_numpy(inst_seg_label).long()

            sample = {'input_tensor': image, 'binary_tensor': bin_seg_label, 'instance_tensor': inst_seg_label,
                      'raw_file':self.image_list[idx]}

            return sample

        elif self.phase == 'test' or 'test_extend':
            '''OpenCV'''
            img_path = os.path.join(self.dataset_dir, self.image_list[idx])
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image -= VGG_MEAN
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float() / 255

            clip, seq, frame = self.image_list[idx].split('/')[-3:]
            path = '/'.join([clip, seq, frame])

            sample = {'input_tensor': image, 'raw_file':self.image_list[idx], 'path':path}

            return sample

        else:
            raise Exception(f"Phase '{self.phase}' cannot be recognize!")




if __name__ == '__main__':
    
    test_set = TuSimpleDataset('/root/Projects/lane_detection/dataset/tusimple/test_set', phase='test')
    train_set = TuSimpleDataset('/root/Projects/lane_detection/dataset/tusimple/train_set', phase='train')
    val_set = TuSimpleDataset('/root/Projects/lane_detection/dataset/tusimple/train_set', phase='val')

    # train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    # valid_loader = DataLoader(valid_set, batch_size=4, shuffle=True, num_workers=1)

    for idx, item in enumerate(train_set):
        input_tensor = item['input_tensor']
        bin_seg_label = item['binary_tensor']
        inst_seg_label = item['instance_tensor']

        input = ((input_tensor * 255).numpy().transpose(1, 2, 0) + np.array(VGG_MEAN)).astype(np.uint8)
        bin_seg_label = (bin_seg_label * 255).numpy().astype(np.uint8)
        inst_seg_label = (inst_seg_label * 50).numpy().astype(np.uint8)

        cv2.imshow('input', input)
        cv2.imshow('bin_seg_label', bin_seg_label)
        cv2.imshow('inst_seg_label', inst_seg_label)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # break


