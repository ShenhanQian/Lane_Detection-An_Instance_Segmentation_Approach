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


class TuSimpleDataset_old(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, info_file, is_training=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.is_training = is_training
        self.transform = transform
        assert os.path.exists(info_file), 'File {} does not exist!'.format(info_file)

        self.image_list = []
        self.binary_list = []
        self.instance_list = []
        # self.partition_list = []
        # self.attract_x_list = []
        # self.attract_y_list = []
        # self.ground_list = []
        # self.drivable_list = []
        with open(info_file, 'r') as f:
            for line in f:
                tmp = line.split()
                self.image_list.append(tmp[0])
                self.binary_list.append(tmp[1])
                self.instance_list.append(tmp[2])
                # self.partition_list.append(tmp[3])
                # self.attract_x_list.append(tmp[4])
                # self.attract_y_list.append(tmp[5])
                # self.ground_list.append(tmp[6])
                # self.drivable_list.append(tmp[3])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        if self.is_training:

            '''OpenCV'''
            # name = self.image_list[idx].split('/')[-1].split('.')[0]

            image = cv2.imread(self.image_list[idx], cv2.IMREAD_COLOR)
            image = np.asarray(image).astype(np.float32)
            image -= VGG_MEAN
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float() /255



            binary = cv2.imread(self.binary_list[idx], cv2.IMREAD_GRAYSCALE) / 255
            binary = torch.from_numpy(binary).long()

            instance = cv2.imread(self.instance_list[idx], cv2.IMREAD_GRAYSCALE)
            instance = torch.from_numpy(instance).long()

            # attract_x = np.load(self.attract_x_list[idx])
            # attract_x = torch.from_numpy(attract_x).float()

            # attract_y = np.load(self.attract_y_list[idx])
            # attract_y = torch.from_numpy(attract_y).float()

            # ground = cv2.imread(self.ground_list[idx], cv2.IMREAD_GRAYSCALE) / 255
            # ground = torch.from_numpy(ground).long()

            # drivable = cv2.imread(self.drivable_list[idx], cv2.IMREAD_UNCHANGED) / 255
            # drivable = torch.from_numpy(drivable).byte()

            # partition = cv2.imread(self.partition_list[idx], cv2.IMREAD_GRAYSCALE)
            # partition = torch.from_numpy(partition).byte()

            # sample = {'name':name, 'input_tensor': image, 'binary_tensor': binary, 'instance_tensor': instance,
            #           'attract_x_tensor': attract_x, 'attract_y_tensor': attract_y}

            sample = {'input_tensor': image, 'binary_tensor': binary, 'instance_tensor': instance}
            # 'name': name, 'ground_tensor': ground, 'partition_tensor': partition}

        else:
            '''OpenCV'''
            image = cv2.imread(self.image_list[idx], cv2.IMREAD_COLOR)
            # image = Image.open(self.image_list[idx])
            image = np.asarray(image).astype(np.float32)
            image -= VGG_MEAN
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

            sample = {'input_tensor': image}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample


class TuSimpleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_dir, phase, size=(512,288), transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
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
        elif phase == 'test_extend':
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
                # print(idx)
                # print(lane)
                cv2.polylines(bin_seg_label, [lane], False, 1, 10)
                cv2.polylines(inst_seg_label, [lane], False, idx+1, 10)
                # cv2.polylines(inst_seg_label, [lane], False, utils.get_color(idx), 10)

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


