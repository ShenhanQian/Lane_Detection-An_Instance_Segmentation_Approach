from dataset import TuSimpleDataset_old, TuSimpleDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import transforms
from model import res_unet
from model.loss import discriminative_loss
import model.lanenet as lanenet
from model.utils import cluster_embed, fit_lanes, sample_from_curve, sample_from_IPMcurve, get_color
# from model.mean_shift import Bin_Mean_Shift
import torch
import time
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np
import cv2
import argparse
import os


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, help='path to parameter file (.pth)')
    parser.add_argument('--tag', type=str, help='training tag')

    return parser.parse_args()


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)


if __name__ == '__main__':

    args = init_args()

    '''Test config'''
    batch_size = 1  # 8G: 14       12G: 18     24G:36
    num_steps = 2000000
    num_workers = 1
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    # writer = SummaryWriter(log_dir='summary/lane-detect-%s-%s' % (train_start_time, args.tag))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size *= torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Let's use CPU")
    print("Batch size: %d" % batch_size)

    output_dir = '/root/Projects/lane_detection/code/lane-detection-pytorch/output/Test-%s-%s' % (train_start_time, args.tag)
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    # data_dir = r'G:\Dataset\tusimple\training_new'
    # data_dir = '/root/Projects/lane_detection/dataset/tusimple/train_set/training_new'
    # data_dir = '/root/Projects/lane_detection/dataset/tusimple/train_set/training_thickness_10'
    data_dir = '/root/Projects/lane_detection/dataset/tusimple/train_set/training_thickness10_2020'
    # data_dir = '/root/Projects/lane_detection/dataset/tusimple/train_set/training_thickness_10-1280x720'
    # data_dir = '/root/Projects/lane_detection/dataset/tusimple/train_set/training_thickness_16'
    # data_dir = '/root/Projects/lane_detection/dataset/tusimple/test_set'

    # train_set = TuSimpleDataset(os.path.join(data_dir, 'train.txt'))
    val_set = TuSimpleDataset(os.path.join(data_dir, 'valid.txt'))
    # test_set = TuSimpleDataset('/root/Projects/lane_detection/dataset/tusimple/test_set', is_training=False)

    num_val = len(val_set)
    # num_test = len(test_set)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valset_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # testset_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {'val': valset_loader}
    phase = 'val'

    print('Finish loading data from %s' % data_dir)

    '''Constant variables'''
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)

    h, w = 288, 512
    src = np.float32([[0.35 * (w - 1), 0.34 * (h - 1)], [0.65 * (w - 1), 0.34 * (h - 1)],
                      [0. * (w - 1), h - 1], [1. * (w - 1), h - 1]])
    dst = np.float32([[0. * (w - 1), 0. * (h - 1)], [1.0 * (w - 1), 0. * (h - 1)],
                      [0.4 * (w - 1), (h - 1)], [0.60 * (w - 1), (h - 1)]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    y_start = np.round(240 * h / 720.)
    y_stop = np.round(710 * h / 720.)
    # y_num = 48
    y_num = 192
    y_sample = np.linspace(y_start, y_stop, y_num)
    x_sample = np.zeros_like(y_sample) + w // 2
    c_sample = np.ones_like(y_sample)
    xyc_sample = np.vstack((x_sample, y_sample, c_sample))

    xyc_IPM = M.dot(xyc_sample).T
    # print(yxc_IPM.T)

    y_IPM = []
    for pt in xyc_IPM:
        y = np.round(pt[1] / pt[2])
        if 0 <= y < h:
            y_IPM.append(y)
    y_IPM = np.array(y_IPM)

    with torch.no_grad():
        # net = ResUNet.ResUNet_1E3D()
        # net = ResUNet.ResUNet_1E1D()
        # net = res_unet.LaneNet()
        net = res_unet.LaneNet_1E2D()
        # net = lanenet.LaneNet_ENet()
        # net = lanenet.ICNet_1E2D()
        #
        net = nn.DataParallel(net)
        net.to(device)
        net.eval()

        assert args.ckpt_path is not None, 'Checkpoint loaded.'

        checkpoint = torch.load(args.ckpt_path)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)  # , strict=False

        step = 0
        epoch = 1

        print()

        sum_bin_precision_train, sum_bin_precision_val = 0, 0
        sum_bin_recall_train, sum_bin_recall_val = 0, 0
        sum_bin_F1_train, sum_bin_F1_val = 0, 0

        # torch.save(net.state_dict(), '%s_epoch-%d.pth' % (train_start_time, epoch))
        data_iter = {'val': iter(dataloaders['val'])}
        time_fp_avg = 0
        time_clst_avg = 0
        time_fit_avg = 0
        time_ct = 0
        for step in range(step, num_steps):
            start_time = time.time()
            time_fp = time.time()

            '''load dataset'''
            try:
                batch = next(data_iter[phase])
            except StopIteration:
                data_iter[phase] = iter(dataloaders[phase])
                batch = next(data_iter[phase])


                avg_precision_bin_val = sum_bin_precision_val / num_val
                avg_recall_bin_val = sum_bin_recall_val / num_val
                avg_F1_bin_val = sum_bin_F1_val / num_val

                # writer.add_scalar('Epoch_Precision_Bin-VAL', avg_precision_bin_val, step)
                # writer.add_scalar('Epoch_Recall_Bin-VAL', avg_recall_bin_val, step)
                # writer.add_scalar('Epoch_F1_Bin-VAL', avg_F1_bin_val, step)

                sum_bin_precision_val = 0
                sum_bin_recall_val = 0
                sum_bin_F1_val = 0

            # names = batch['name']
            input_batch = batch['input_tensor']
            labels_bin_batch = batch['binary_tensor']
            labels_inst_batch = batch['instance_tensor']

            input_batch = input_batch.to(device)
            labels_bin_batch = labels_bin_batch.to(device)
            labels_inst_batch = labels_inst_batch.to(device)

            # forward
            embeddings, logit = net(input_batch)

            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)
            preds_bin_expand_batch = pred_bin_batch.view(pred_bin_batch.shape[0] * pred_bin_batch.shape[1] * pred_bin_batch.shape[2] * pred_bin_batch.shape[3])
            labels_bin_expand_batch = labels_bin_batch.view(labels_bin_batch.shape[0] * labels_bin_batch.shape[1] * labels_bin_batch.shape[2])

            time_fp = time.time() - time_fp

            '''sklearn mean_shift'''
            time_clst = time.time()
            pred_insts = cluster_embed(embeddings, pred_bin_batch, band_width=0.5)
            time_clst = time.time() - time_clst

            time_fit = time.time()
            _, h, w = pred_insts.shape

            for idx in range(batch_size):
                input_rgb = input_batch[idx]  # for each image in a batch
                pred_inst = pred_insts[idx]

                '''IPM'''
                pred_inst_IPM = cv2.warpPerspective(pred_inst.cpu().numpy().astype('uint8'), M, (w, h),
                                                    flags=cv2.INTER_NEAREST)
                pred_inst_IPM = torch.from_numpy(pred_inst_IPM)
                curves_param = fit_lanes(pred_inst_IPM)
                curves_pts_IPM = sample_from_IPMcurve(curves_param, pred_inst, y_IPM)

                curves_pts_pred = []
                for xy_IPM in curves_pts_IPM:  # for each lane in a image
                    n, _ = xy_IPM.shape
                    # y_sample = np.linspace(y_start, y_stop, 48)  # 16
                    # x_sample = np.zeros_like(y_sample) + w // 2
                    c_IPM = np.ones((n,1))
                    xyc_IPM = np.hstack((xy_IPM, c_IPM))
                    xyc_pred = M_inv.dot(xyc_IPM.T).T

                    xy_pred = []
                    for pt in xyc_pred:
                        x = np.round(pt[0] / pt[2]).astype(np.uint16)
                        y = np.round(pt[1] / pt[2]).astype(np.uint16)
                        if 0 <= y < h and 0 <= x < w:  #  and pred_inst[y, x]
                            xy_pred.append([x,y])
                    xy_pred = np.array(xy_pred, dtype=np.uint16)

                    curves_pts_pred.append(xy_pred)

                '''origin'''
                # curves_param = fit_lanes(pred_inst)
                # curves_pts=sample_from_curve(curves_param,pred_inst)

                '''visualize'''
                curve_sample = np.zeros((h, w, 3), dtype=np.uint8)
                curve_sample_IPM = np.zeros((h, w, 3), dtype=np.uint8)
                rgb = ((input_rgb).cpu().numpy().transpose(1, 2, 0) * 255 + VGG_MEAN).astype(np.uint8)
                rgb_IPM = cv2.warpPerspective(rgb, M, (w, h),
                                                    flags=cv2.INTER_LINEAR)
                pred_bin_rgb = pred_bin_batch[idx].repeat(3,1,1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 255
                pred_inst_rgb = pred_inst.repeat(3,1,1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 40
                pred_inst_IPM_rgb = pred_inst_IPM.repeat(3,1,1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 40

                for idx, inst in enumerate(curves_pts_pred):
                    if inst.ndim == 2:
                        pts = inst.transpose((1, 0))
                        curve_sample[pts[1], pts[0]] = (0, 0, 255)
                        rgb[pts[1], pts[0]] = (0, 0, 255)
                        pred_bin_rgb[pts[1], pts[0]] = (0, 0, 255)
                        pred_inst_rgb[pts[1], pts[0]] = (0, 0, 255)
                        # input('Ctrl-c')

                for idx, inst in enumerate(curves_pts_IPM):
                    pts = inst.transpose((1, 0))
                    curve_sample_IPM[pts[1], pts[0]] = (0, 0, 255)
                    rgb_IPM[pts[1], pts[0]] = (0, 0, 255)
                    pred_inst_IPM_rgb[pts[1], pts[0]] = (0, 0, 255)

                # align_IPM=cv2.warpPerspective(align,M,(w,h))
                # pred_inst_IPM=cv2.warpPerspective(pred_inst.cpu().numpy().astype(np.uint8),M,(w,h))

                # # inst_lbl_rgb=np.repeat(inst_lbl.cpu().numpy().astype(np.uint8).reshape(h,w,1),3,2)
                #
                # # inst_lbl_rgb=cv2.addWeighted(align,0.5,inst_lbl_rgb,0.5,0)
                #

                cv2.imshow('input',rgb)
                cv2.imshow('input_IPM',rgb_IPM)
                cv2.imshow('inst_pred_IPM',pred_inst_IPM_rgb)
                cv2.imshow('inst_pred',pred_inst_rgb)
                cv2.imshow('curve', curve_sample)
                cv2.imshow('curve_IPM', curve_sample_IPM)

                cv2.waitKey(0)
            time_fit = time.time() - time_fit


            '''gpu Mean Shift'''
            # # fast mean shift
            # bin_mean_shift = Bin_Mean_Shift()
            # segmentation = bin_mean_shift.test_forward(pred_bin_batch[0], embeddings[0], mask_threshold=0.1)

            # statistics
            # bin_corrects = torch.sum((preds_bin_expand_batch.detach() == labels_bin_expand_batch.detach()).byte())

            bin_TP = torch.sum((preds_bin_expand_batch.detach() == labels_bin_expand_batch.detach()) & (preds_bin_expand_batch.detach() == 1))
            bin_precision = bin_TP.double() / (torch.sum(preds_bin_expand_batch.detach() == 1).double() + 1e-6)
            bin_recall = bin_TP.double() / (torch.sum(labels_bin_expand_batch.detach() == 1).double() + 1e-6)
            bin_F1 = 2 * bin_precision * bin_recall / (bin_precision + bin_recall)

            step_time = time.time() - start_time

            # print(pred_bin_batch.shape[0])
            sum_bin_precision_val += bin_precision.detach() * pred_bin_batch.shape[0]
            sum_bin_recall_val += bin_recall.detach() * pred_bin_batch.shape[0]
            sum_bin_F1_val += bin_F1.detach() * pred_bin_batch.shape[0]

            # writer.add_scalar('total_val_loss', loss.item(), step)
            # writer.add_scalar('bin_val_loss', loss_bin.item(), step)
            # writer.add_scalar('bin_val_F1', bin_F1, step)
            # writer.add_scalar('disc_val_loss', loss_disc.item(), step)

            if step > 10:
                time_fp_avg = (time_ct * time_fp_avg + time_fp) / (time_ct + 1)
                time_clst_avg = (time_ct * time_clst_avg + time_clst) / (time_ct + 1)
                time_fit_avg = (time_ct * time_fit_avg + time_fit) / (time_ct + 1)

                time_ct += 1
            print('{}  {}  Epoch:{}  Step:{}  '
                  'BinRecall:{:.5f}  BinPrec:{:.5f}  F1:{:.5f}  '
                  'Time:{:.2f}  time_fp_avg:{:.1f}  time_clst_avg:{:.1f}  time_fit_avg:{:.1f}'
                  .format(train_start_time, args.tag, epoch, step,
                          bin_recall.item(), bin_precision.item(), bin_F1.item(),
                          step_time, time_fp_avg * 1000, time_clst_avg * 1000, time_fit_avg * 1000))

            '''Visualization'''

            # num_images = 3
            #
            # inputs_images = (input_batch + VGG_MEAN)[:num_images, [2, 1, 0], :, :]  # .byte()
            # writer.add_images('image', inputs_images, step)
            #
            # writer.add_images('Bin Pred', pred_bin_batch[:num_images], step)
            #
            # labels_bin_img = labels_bin_batch.view(labels_bin_batch.shape[0], 1, labels_bin_batch.shape[1], labels_bin_batch.shape[2])
            # writer.add_images('Bin Label', labels_bin_img[:num_images], step)
            #
            # embedding_img = F.normalize(embeddings[:num_images], 1, 1) / 2. + 0.5
            # # print(torch.min(embedding_img).item(), torch.max(embedding_img).item())
            # writer.add_images('Embedding', embedding_img, step)






