import os
import time
import argparse
import ujson as json
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import model.lanenet as lanenet
from model.utils import cluster_embed, fit_lanes, sample_from_curve, sample_from_IPMcurve, generate_json_entry, get_color
from dataset import TuSimpleDataset


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to TuSimple Benchmark dataset')
    parser.add_argument('--ckpt_path', type=str, help='path to parameter file (.pth)')
    parser.add_argument('--arch', type=str, default='fcn', help='network architecture type(default: FCN)')
    parser.add_argument('--dual_decoder', action='store_true', help='use seperate decoders for two branches')
    parser.add_argument('--ipm', action='store_true', help='whether to perform Inverse Projective Mapping(IPM) before curve fitting')
    parser.add_argument('--show', action='store_true', help='whether to show visualization images when testing')
    parser.add_argument('--save_img', action='store_true', help='whether to save visualization images when testing')
    parser.add_argument('--tag', type=str, help='tag to log details of experiments')

    return parser.parse_args()


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)


if __name__ == '__main__':

    args = init_args()

    '''Test config'''
    batch_size = 1
    num_workers = 4
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    # writer = SummaryWriter(log_dir='summary/lane-detect-%s-%s' % (train_start_time, args.tag))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size *= torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device("cpu")
        print("Let's use CPU")
    print("Batch size: %d" % batch_size)

    output_dir = './output/Test-%s-%s' % (train_start_time, args.tag)
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    data_dir = args.data_dir

    test_set = TuSimpleDataset(data_dir, phase='test')
    # test_set = TuSimpleDataset('/root/Projects/lane_detection/dataset/tusimple/test_set', phase='test_extend')
    # val_set = TuSimpleDataset('/root/Projects/lane_detection/dataset/tusimple/train_set', phase='val')

    num_test = len(test_set)

    testset_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {'test': testset_loader}
    phase = 'test'

    print('Finish loading data from %s' % data_dir)

    '''Constant variables'''
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)

    _, h, w = test_set[0]['input_tensor'].shape

    # for IPM (Inverse Projective Mapping)
    src = np.float32([[0.35 * (w - 1), 0.34 * (h - 1)], [0.65 * (w - 1), 0.34 * (h - 1)],
                      [0. * (w - 1), h - 1], [1. * (w - 1), h - 1]])
    dst = np.float32([[0. * (w - 1), 0. * (h - 1)], [1.0 * (w - 1), 0. * (h - 1)],
                      [0.4 * (w - 1), (h - 1)], [0.60 * (w - 1), (h - 1)]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    # y_start, y_stop and y_num is calculated according to TuSimple Benchmark's setting
    y_start = np.round(160 * h / 720.)
    y_stop = np.round(710 * h / 720.)
    y_num = 56
    y_sample = np.linspace(y_start, y_stop, y_num, dtype=np.int16)
    x_sample = np.zeros_like(y_sample, dtype=np.float32) + w // 2
    c_sample = np.ones_like(y_sample, dtype=np.float32)
    xyc_sample = np.vstack((x_sample, y_sample, c_sample))

    xyc_IPM = M.dot(xyc_sample).T

    y_IPM = []
    for pt in xyc_IPM:
        y = np.round(pt[1] / pt[2])
        y_IPM.append(y)
    y_IPM = np.array(y_IPM)

    '''Forward propogation'''
    with torch.no_grad():
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
        net.eval()

        assert args.ckpt_path is not None, 'Checkpoint Error.'

        checkpoint = torch.load(args.ckpt_path)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)

        step = 0
        epoch = 1
        print()

        data_iter = {'test': iter(dataloaders['test'])}
        time_run_avg = 0
        time_fp_avg = 0
        time_clst_avg = 0
        time_fit_avg = 0
        time_ct = 0
        output_list = list()
        for step in range(num_test):
            time_run = time.time()
            time_fp = time.time()

            '''load dataset'''
            try:
                batch = next(data_iter[phase])
            except StopIteration:
                break

            input_batch = batch['input_tensor']
            raw_file_batch = batch['raw_file']
            path_batch = batch['path']

            input_batch = input_batch.to(device)

            # forward
            embeddings, logit = net(input_batch)

            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)
            preds_bin_expand_batch = pred_bin_batch.view(pred_bin_batch.shape[0] * pred_bin_batch.shape[1] * pred_bin_batch.shape[2] * pred_bin_batch.shape[3])

            time_fp = time.time() - time_fp

            '''sklearn mean_shift'''
            time_clst = time.time()
            pred_insts = cluster_embed(embeddings, pred_bin_batch, band_width=0.5)
            time_clst = time.time() - time_clst

            '''Curve Fitting'''
            time_fit = time.time()
            for idx in range(batch_size):
                input_rgb = input_batch[idx]  # for each image in a batch
                raw_file = raw_file_batch[idx]
                pred_inst = pred_insts[idx]
                path = path_batch[idx]
                if args.ipm:
                    '''Fit Curve after IPM(Inverse Perspective Mapping)'''
                    pred_inst_IPM = cv2.warpPerspective(pred_inst.cpu().numpy().astype('uint8'), M, (w, h),
                                                        flags=cv2.INTER_NEAREST)
                    pred_inst_IPM = torch.from_numpy(pred_inst_IPM)

                    curves_param = fit_lanes(pred_inst_IPM)
                    curves_pts_IPM = sample_from_IPMcurve(curves_param, pred_inst_IPM, y_IPM)
                    
                    curves_pts_pred = []
                    for xy_IPM in curves_pts_IPM:  # for each lane in a image
                        n, _ = xy_IPM.shape
                    
                        c_IPM = np.ones((n, 1))
                        xyc_IPM = np.hstack((xy_IPM, c_IPM))
                        xyc_pred = M_inv.dot(xyc_IPM.T).T
                    
                        xy_pred = []
                        for pt in xyc_pred:
                            x = np.round(pt[0] / pt[2]).astype(np.int32)
                            y = np.round(pt[1] / pt[2]).astype(np.int32)
                            if 0 <= y < h and 0 <= x < w:  # and pred_inst[y, x]
                                xy_pred.append([x,y])
                            else:
                                xy_pred.append([-2, y])
                    
                        xy_pred = np.array(xy_pred, dtype=np.int32)
                        curves_pts_pred.append(xy_pred)
                else:
                    '''Directly fit curves on original images'''
                    curves_param = fit_lanes(pred_inst)
                    curves_pts_pred=sample_from_curve(curves_param,pred_inst, y_sample)

                '''Visualization'''
                curve_sample = np.zeros((h, w, 3), dtype=np.uint8)
                rgb = (input_rgb.cpu().numpy().transpose(1, 2, 0) * 255 + VGG_MEAN).astype(np.uint8)
                pred_bin_rgb = pred_bin_batch[idx].repeat(3,1,1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 255
                # pred_inst_rgb = pred_inst.repeat(3,1,1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 40  # gray
                pred_inst_rgb = pred_inst.repeat(3, 1, 1).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)  # color
                
                for i in np.unique(pred_inst_rgb):
                    if i == 0:
                        continue
                    index = np.where(pred_inst_rgb[:, :, 0] == i)
                    pred_inst_rgb[index] = get_color(i)

                fg_mask = (pred_bin_rgb[:, :, 0] == 255).astype(np.uint8)
                bg_mask = (pred_bin_rgb[:, :, 0] == 0).astype(np.uint8)
                rgb_bg = cv2.bitwise_and(rgb, rgb, mask=bg_mask)

                rgb_fg = cv2.bitwise_and(rgb, rgb, mask=fg_mask)
                pred_inst_rgb_fg = cv2.bitwise_and(pred_inst_rgb, pred_inst_rgb, mask=fg_mask)
                fg_align = cv2.addWeighted(rgb_fg, 0.3, pred_inst_rgb_fg, 0.7, 0)
                rgb_align = rgb_bg + fg_align

                if args.save_img:
                    clip, seq, frame = path.split('/')
                    output_seq_dir = os.path.join(output_dir, seq)
                    if os.path.exists(output_seq_dir) is False:
                        os.makedirs(output_seq_dir, exist_ok=True)

                    cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_input.jpg'), rgb)
                    cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_bin_pred.jpg'), pred_bin_rgb)
                    cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_inst_pred.jpg'), pred_inst_rgb)
                    cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_align.jpg'), rgb_align)


                if args.ipm:
                    curve_sample_IPM = np.zeros((h, w, 3), dtype=np.uint8)
                    rgb_IPM = cv2.warpPerspective(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
                    pred_inst_IPM_rgb = pred_inst_IPM.repeat(3,1,1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 40

                if args.show:
                    if args.ipm:
                        '''for IPM image'''
                        for idx, inst in enumerate(curves_pts_IPM):
                            m = np.logical_and(0 <= inst[:, 0], inst[:, 0] < w)
                            m = np.logical_and(m, inst[:, 1] >= y_start)
                            m = np.logical_and(m, inst[:, 1] <= y_stop)
                            index = np.nonzero(m)
                            inst = inst[index]

                            pts = inst.transpose((1, 0))
                            curve_sample_IPM[pts[1], pts[0]] = (0, 0, 255)
                            rgb_IPM[pts[1], pts[0]] = (0, 0, 255)
                            pred_inst_IPM_rgb[pts[1], pts[0]] = (0, 0, 255)
                        
                            cv2.polylines(curve_sample_IPM, [inst.astype(np.int32)], False, (0, 0, 255), 2)
                            cv2.polylines(rgb_IPM, [inst.astype(np.int32)], False, (0, 0, 255), 2)
                            cv2.polylines(pred_inst_IPM_rgb, [inst.astype(np.int32)], False, (0, 0, 255), 2)
                        cv2.imshow('input_IPM', rgb_IPM)
                        cv2.imshow('inst_pred_IPM', pred_inst_IPM_rgb)
                        cv2.imshow('curve_IPM', curve_sample_IPM)

                    else:
                        '''for front-face view image'''
                        for idx, inst in enumerate(curves_pts_pred):
                            if inst.ndim == 2:
                                index = np.nonzero(inst[:, 0] != -2)
                                inst = inst[index]
                        
                                pts = inst.transpose((1, 0))
                                curve_sample[pts[1], pts[0]] = (0, 0, 255)
                                rgb[pts[1], pts[0]] = (0, 0, 255)
                                pred_bin_rgb[pts[1], pts[0]] = (0, 0, 255)
                                pred_inst_rgb[pts[1], pts[0]] = (0, 0, 255)
                        
                                cv2.polylines(rgb, [inst.astype(np.int32)], False, (0, 0, 255), 2)
                                cv2.polylines(pred_bin_rgb, [inst.astype(np.int32)], False, (0, 0, 255), 2)
                                cv2.polylines(pred_inst_rgb, [inst.astype(np.int32)], False, (0, 0, 255), 2)
                        
                        cv2.imshow('inst_pred', pred_inst_rgb)
                        cv2.imshow('curve', curve_sample)
                        cv2.imshow('align', rgb_align)
                        # cv2.imshow('input',rgb)
                        # cv2.imshow('bin_pred', pred_bin_rgb)

                    cv2.waitKey(0)  # wait forever until a key stroke
                    # cv2.waitKey(1)  # wait for 1ms

                time_fit = time.time() - time_fit
                time_run = time.time() - time_run

                # Generate Json file to be evaluated by TuSimple Benchmark official eval script
                json_entry = generate_json_entry(curves_pts_pred, y_sample, raw_file, (h, w), time_run)
                output_list.append(json_entry)

                time_run_avg = (time_ct * time_run_avg + time_run) / (time_ct + 1)
                time_fp_avg = (time_ct * time_fp_avg + time_fp) / (time_ct + 1)
                time_clst_avg = (time_ct * time_clst_avg + time_clst) / (time_ct + 1)
                time_fit_avg = (time_ct * time_fit_avg + time_fit) / (time_ct + 1)
                time_ct += 1

                if step % 50 == 0:  # Change the coefficient to filter the value
                    time_ct = 0


                print('{}  {}  Epoch:{}  Step:{}  Time:{:5.1f}  '
                      'time_run_avg:{:5.1f}  time_fp_avg:{:5.1f}  time_clst_avg:{:5.1f}  time_fit_avg:{:5.1f}  fps_avg:{:d}'
                      .format(train_start_time, args.tag, epoch, step, time_run*1000,
                              time_run_avg*1000, time_fp_avg * 1000, time_clst_avg * 1000, time_fit_avg * 1000,
                              int(1/(time_run_avg + 1e-9))))

            '''Write to Tensorboard Summary'''
            # num_images = 3
            # inputs_images = (input_batch + VGG_MEAN)[:num_images, [2, 1, 0], :, :]  # .byte()
            # writer.add_images('image', inputs_images, step)
            # #
            # writer.add_images('Bin Pred', pred_bin_batch[:num_images], step)
            # #
            # labels_bin_img = labels_bin_batch.view(labels_bin_batch.shape[0], 1, labels_bin_batch.shape[1], labels_bin_batch.shape[2])
            # writer.add_images('Bin Label', labels_bin_img[:num_images], step)
            #
            # embedding_img = F.normalize(embeddings[:num_images], 1, 1) / 2. + 0.5
            # # print(torch.min(embedding_img).item(), torch.max(embedding_img).item())
            # writer.add_images('Embedding', embedding_img, step)

        with open(f'output/test_pred-{train_start_time}-{args.tag}.json', 'w') as f:
            for item in output_list:
                json.dump(item, f)  # , indent=4, sort_keys=True
                f.write('\n')

