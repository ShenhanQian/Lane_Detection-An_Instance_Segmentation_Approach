import argparse
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from model.loss import discriminative_loss
import model.lanenet as lanenet
from dataset import TuSimpleDataset


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to TuSimple Benchmark dataset')
    parser.add_argument('--ckpt_path', type=str, help='path to parameter file (.pth)')
    parser.add_argument('--arch', type=str, default='fcn', help='network architecture type(default: FCN)')
    parser.add_argument('--dual_decoder', action='store_true', help='use seperate decoders for two branches')
    parser.add_argument('--tag', type=str, help='training tag')

    return parser.parse_args()


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)


if __name__ == '__main__':

    args = init_args()

    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)
    VGG_MEAN = torch.from_numpy(VGG_MEAN).cuda().view([1, 3, 1, 1])
    batch_size = 16  # batch size per GPU
    learning_rate = 1e-3  # 1e-3
    num_steps = 2000000
    num_workers = 4
    ckpt_epoch_interval = 10  # save a model checkpoint every X epochs
    val_step_interval = 50  # perform a validation step every X traning steps
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size *= torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Let's use CPU")
    print("Batch size: %d" % batch_size)

    data_dir = args.data_dir
    train_set = TuSimpleDataset(data_dir, 'train')
    val_set = TuSimpleDataset(data_dir, 'val')

    num_train = len(train_set)
    num_val = len(val_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders = {'train': train_loader, 'val': val_loader}
    print('Finish loading data from %s' % data_dir)

    writer = SummaryWriter(log_dir='summary/lane-detect-%s-%s' % (train_start_time, args.tag))

    # select NN architecture
    arch = args.arch
    if 'fcn' in arch.lower():
        arch = 'lanenet.LaneNet_FCN_Res'
    elif 'enet' in arch.lower():
        arch = 'lanenet.LaneNet_ENet'
    elif 'icnet' in arch.lower():
        arch = 'lanenet.LaneNet_ICNet'
    
    arch = arch + '_1E2D' if args.dual_decoder else arch + '_1E1D'
    print('Architecture:', arch)
    net = eval(arch)()
    
    # net = lanenet.LaneNet_FCN_Res_1E1D()
    # net = lanenet.LaneNet_FCN_Res_1E2D()
    # net = lanenet.LaneNet_ENet_1E1D()
    # net = lanenet.LaneNet_ENet_1E2D()
    # net = lanenet.LaneNet_ICNet_1E2D()
    # net = lanenet.LaneNet_ICNet_1E1D()

    net = nn.DataParallel(net)
    net.to(device)

    params_to_update = net.parameters()
    # for name, param in net.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t", name)
    # optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(params_to_update)
    MSELoss = nn.MSELoss()

    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path)
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)  # , strict=False
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
        #step = checkpoint['step']
        step = 0  # by default, we reset step and epoch value
        epoch = 1

        loss = checkpoint['loss']
        print('Checkpoint loaded.')

    else:
        net.apply(init_weights)
        step = 0
        epoch = 1

        print('Network parameters initialized.')
    
    # accumulators to calculate statistics in each epoch 
    sum_bin_precision_train, sum_bin_precision_val = 0, 0
    sum_bin_recall_train, sum_bin_recall_val = 0, 0
    sum_bin_F1_train, sum_bin_F1_val = 0, 0

    '''session'''
    data_iter = {'train': iter(dataloaders['train']), 'val': iter(dataloaders['val'])}
    for step in range(step, num_steps):
        start_time = time.time()

        phase = 'train'
        net.train()
        if step % val_step_interval == 0:
            phase = 'val'
            net.eval()

        '''load dataset'''
        try:
            batch = next(data_iter[phase])
        except StopIteration:
            data_iter[phase] = iter(dataloaders[phase])
            batch = next(data_iter[phase])

            if phase == 'train':
                epoch += 1
                if epoch % ckpt_epoch_interval == 0:
                    ckpt_dir = 'check_point/ckpt_%s_%s' % (train_start_time, args.tag)
                    if os.path.exists(ckpt_dir) is False:
                        os.mkdir(ckpt_dir)
                    ckpt_path = os.path.join(ckpt_dir, 'ckpt_%s_epoch-%d.pth' % (train_start_time, epoch))
                    torch.save({
                        'epoch': epoch,
                        'step': step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, ckpt_path)

                avg_precision_bin_train = sum_bin_precision_train / num_train
                avg_recall_bin_train = sum_bin_recall_train / num_train
                avg_F1_bin_train = sum_bin_F1_train / num_train
                writer.add_scalar('Epoch_Precision_Bin-TRAIN', avg_precision_bin_train, step)
                writer.add_scalar('Epoch_Recall_Bin-TRAIN', avg_recall_bin_train, step)
                writer.add_scalar('Epoch_F1_Bin-TRAIN', avg_F1_bin_train, step)
                writer.add_text('Epoch', str(epoch), step)
                sum_bin_precision_train = 0
                sum_bin_recall_train = 0
                sum_bin_F1_train = 0

            elif phase == 'val':
                avg_precision_bin_val = sum_bin_precision_val / num_val
                avg_recall_bin_val = sum_bin_recall_val / num_val
                avg_F1_bin_val = sum_bin_F1_val / num_val
                writer.add_scalar('Epoch_Precision_Bin-VAL', avg_precision_bin_val, step)
                writer.add_scalar('Epoch_Recall_Bin-VAL', avg_recall_bin_val, step)
                writer.add_scalar('Epoch_F1_Bin-VAL', avg_F1_bin_val, step)
                sum_bin_precision_val = 0
                sum_bin_recall_val = 0
                sum_bin_F1_val = 0

        inputs = batch['input_tensor']
        labels_bin = batch['binary_tensor']
        labels_inst = batch['instance_tensor']

        inputs = inputs.to(device)
        labels_bin = labels_bin.to(device)
        labels_inst = labels_inst.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        embeddings, logit = net(inputs)

        # compute loss
        preds_bin = torch.argmax(logit, dim=1, keepdim=True)
        preds_bin_expand = preds_bin.view(preds_bin.shape[0] * preds_bin.shape[1] * preds_bin.shape[2] * preds_bin.shape[3])
        labels_bin_expand = labels_bin.view(labels_bin.shape[0] * labels_bin.shape[1] * labels_bin.shape[2])

        '''Floating Loss weighting determined by label proportion'''
        bin_count = torch.bincount(labels_bin_expand)
        bin_prop = bin_count.float() / torch.sum(bin_count)
        weight_bin = torch.tensor(1) / (bin_prop + 0.2)  # max proportion: 5:1
        # weight_bin = 1. / torch.log(bin_prop + 1.02)  # max proportion: 50:1

        '''Fixed loss weighting'''
        # weight_bin = torch.tensor([1, 5], dtype=torch.float).to(device)

        # binary segmentation loss
        '''Multi-class CE Loss'''
        CrossEntropyLoss = nn.CrossEntropyLoss(weight=weight_bin)
        loss_bin = CrossEntropyLoss(logit, labels_bin)

        # discriminative loss
        loss_disc, loss_v, loss_d, loss_r = discriminative_loss(embeddings,
                                                                labels_inst,
                                                                delta_v=0.2,
                                                                delta_d=1,
                                                                param_var=.5,
                                                                param_dist=.5,
                                                                param_reg=0.001)

        loss = loss_bin + loss_disc * 0.01

        # backward + optimize only if in training phase
        if phase == 'train':
            loss.backward()
            optimizer.step()

        # Statistics
        bin_TP = torch.sum((preds_bin_expand.detach() == labels_bin_expand.detach()) & (preds_bin_expand.detach() == 1))
        bin_precision = bin_TP.double() / (torch.sum(preds_bin_expand.detach() == 1).double() + 1e-6)
        bin_recall = bin_TP.double() / (torch.sum(labels_bin_expand.detach() == 1).double() + 1e-6)
        bin_F1 = 2 * bin_precision * bin_recall / (bin_precision + bin_recall)

        step_time = time.time() - start_time
        if phase == 'train':
            step += 1

            sum_bin_precision_train += bin_precision.detach() * preds_bin.shape[0]
            sum_bin_recall_train += bin_recall.detach() * preds_bin.shape[0]
            sum_bin_F1_train += bin_F1.detach() * preds_bin.shape[0]

            writer.add_scalar('learning_rate', learning_rate, step)

            writer.add_scalar('total_train_loss', loss.item(), step)
            writer.add_scalar('bin_train_loss', loss_bin.item(), step)
            writer.add_scalar('bin_train_F1', bin_F1, step)
            writer.add_scalar('disc_train_loss', loss_disc.item(), step)

            print('{}  {}  \nEpoch:{}  Step:{}  TrainLoss:{:.5f}  Bin_Loss:{:.5f}  '
                  'BinRecall:{:.5f}  BinPrec:{:.5f}  F1:{:.5f}  '
                  'DiscLoss:{:.5f}  vLoss:{:.5f}  dLoss:{:.5f}  rLoss:{:.5f}  '
                  'Time:{:.2f}'
                  .format(train_start_time, args.tag, epoch, step, loss.item(), loss_bin.item(),
                          bin_recall.item(), bin_precision.item(), bin_F1.item(),
                          loss_disc.item(), loss_v.item(), loss_d.item(), loss_r.item(),
                          step_time))

        elif phase == 'val':
            sum_bin_precision_val += bin_precision.detach() * preds_bin.shape[0]
            sum_bin_recall_val += bin_recall.detach() * preds_bin.shape[0]
            sum_bin_F1_val += bin_F1.detach() * preds_bin.shape[0]

            writer.add_scalar('total_val_loss', loss.item(), step)
            writer.add_scalar('bin_val_loss', loss_bin.item(), step)
            writer.add_scalar('bin_val_F1', bin_F1, step)
            writer.add_scalar('disc_val_loss', loss_disc.item(), step)

            print('\n{}  {}  \nEpoch:{}  Step:{}  ValidLoss:{:.5f}  BinLoss:{:.5f}  '
                  'BinRecall:{:.5f}  BinPrec:{:.5f}  F1:{:.5f}  '
                  'DiscLoss:{:.5f}  vLoss:{:.5f}  dLoss:{:.5f}  rLoss:{:.5f}  '
                  'Time:{:.2f}'
                  .format(train_start_time, args.tag, epoch, step, loss.item(), loss_bin.item(),
                          bin_recall.item(), bin_precision.item(), bin_F1.item(),
                          loss_disc.item(), loss_v.item(), loss_d.item(), loss_r.item(),
                          step_time))

            '''Save images into Tensorflow summary'''

            num_images = 3  # Select the number of images to be saved in each val iteration

            inputs_images = (inputs + VGG_MEAN / 255.)[:num_images, [2, 1, 0], :, :]
            writer.add_images('image', inputs_images, step)

            writer.add_images('Bin Pred', preds_bin[:num_images], step)

            labels_bin_img = labels_bin.view(labels_bin.shape[0], 1, labels_bin.shape[1], labels_bin.shape[2])
            writer.add_images('Bin Label', labels_bin_img[:num_images], step)

            embedding_img = F.normalize(embeddings[:num_images], 1, 1) / 2. + 0.5  # a tricky way to visualize the embedding
            writer.add_images('Embedding', embedding_img, step)

            # # Embeddings can be saved and viewed by tensorboard, but the process is computationally costly
            # # select embedding pixels with inst_fg_mask
            # embedding = embeddings[0]
            # c, h, w = embedding.shape
            # inst_fg_mask = labels_inst[idx] != 0  # .to(device)
            # embeddings_fg = torch.transpose(torch.masked_select(embedding, inst_fg_mask).view(c, -1), 0, 1)
            # label_inst_fg = torch.masked_select(labels_inst[idx], inst_fg_mask)
            # metadata = [str(i.item()) for i in label_inst_fg]
            # writer.add_embedding(
            #     embeddings_fg,
            #     metadata=label_inst_fg,
            #     # label_img=embedding.detach(),
            #     global_step=step)




