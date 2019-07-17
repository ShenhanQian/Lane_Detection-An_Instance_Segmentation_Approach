#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/10 19:07
# @Author  : Shenhan Qian
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

import numpy as np
import cv2 as cv2
import torch
from sklearn.cluster import MeanShift


def fit_lanes(inst_pred):
    """

    :param inst_pred: lane instances prediction map, support single image
    :return: A list of each curve's parameter
    """
    assert inst_pred.dim() == 2

    h, w = inst_pred.shape


    inst_pred_expand = inst_pred.view(-1)

    inst_unique = torch.unique(inst_pred_expand)

    # extract points coordinates for each lane
    lanes = []
    for inst_idx in inst_unique:
        if inst_idx != 0:

            # lanes.append(torch.nonzero(torch.tensor(inst_pred == inst_idx).byte()).numpy())

            lanes.append(torch.nonzero(inst_pred == inst_idx).cpu().numpy())

    curves = []
    for lane in lanes:
        pts = lane

        # fitting each lane
        curve = np.polyfit(pts[:, 0], pts[:, 1], 3)
        curves.append(curve)

    return curves


def sample_from_curve(curves, inst_pred, y_sample):
    """

    :param curves: A list of each curve's parameter
    :param inst_pred: lane instances prediction map, support single image
    :return: A list of sampled points on each curve
    """
    h, w = inst_pred.shape
    curves_pts = []
    for param in curves:
        # use new curve function f(y) to calculate x values
        fy = np.poly1d(param)
        x_sample = fy(y_sample)

        '''Filter out points beyond image boundaries'''
        index = np.where(np.logical_or(x_sample < 0, x_sample >= w))
        x_sample[index] = -2

        '''Filter out points beyond predictions'''
        # may filter out bad point, but can also drop good point at the edge
        index = np.where((inst_pred[y_sample, x_sample] == 0).cpu().numpy())
        x_sample[index] = -2

        xy_sample = np.vstack((x_sample, y_sample)).transpose((1, 0)).astype(np.int32)

        curves_pts.append(xy_sample)

    return curves_pts


def sample_from_IPMcurve(curves, pred_inst_IPM, y_sample):
    """

    :param curves: A list of each curve's parameter
    :param inst_pred: lane instances prediction map, support single image
    :return: A list of sampled points on each curve
    """
    h, w = pred_inst_IPM.shape
    curves_pts = []
    for param in curves:
        # use new curve function f(y) to calculate x values
        fy = np.poly1d(param)
        x_sample = fy(y_sample)

        xy_sample = np.vstack((x_sample, y_sample)).transpose((1, 0)).astype(np.int32)

        curves_pts.append(xy_sample)

    return curves_pts


def generate_json_entry(curves_pts_pred, y_sample, raw_file, size, run_time):
    h, w = size

    lanes = []
    for curve in curves_pts_pred:
        index = np.where(curve[:, 0] > 0)
        curve[index, 0] = curve[index, 0] * 720. / h

        x_list = np.round(curve[:, 0]).astype(np.int32).tolist()
        lanes.append(x_list)

    entry_dict = dict()

    entry_dict['lanes'] = lanes
    entry_dict['h_sample'] = np.round(y_sample * 720. / h).astype(np.int32).tolist()
    entry_dict['run_time'] = int(np.round(run_time * 1000))
    entry_dict['raw_file'] = raw_file

    return entry_dict

def cluster_embed(embeddings, preds_bin, band_width):
    c = embeddings.shape[1]
    n, _, h, w = preds_bin.shape
    preds_bin = preds_bin.view(n, h, w)
    preds_inst = torch.zeros_like(preds_bin)
    for idx, (embedding, bin_pred) in enumerate(zip(embeddings, preds_bin)):
        # print(embedding.size(), bin_pred.size())
        embedding_fg = torch.transpose(torch.masked_select(embedding, bin_pred.byte()).view(c, -1), 0, 1)
        # print(embedding_fg.size())

        # embedding_expand = embedding.view(embedding.shape[0],
        #                                   embedding.shape[1] * embedding.shape[2])
        # embedding_expand =torch.transpose(embedding_expand, 1, 0)
        # print(embedding_expand.shape)
        clustering = MeanShift(bandwidth=band_width, bin_seeding=True, min_bin_freq=100).fit(embedding_fg.cpu().detach().numpy())

        preds_inst[idx][bin_pred.byte()] = torch.from_numpy(clustering.labels_).cuda() + 1

        # labels_color = get_color(clustering.labels_)
        # preds_inst[idx][bin_pred.byte()] = torch.from_numpy(labels_color).cuda() + 1

        # print(torch.unique(preds_inst[idx]))
    return preds_inst


color_set = [(0, 0, 0),
    (60, 76, 231), (18, 156, 243), (113, 204, 46), (219, 152, 52), (182, 89, 155),
    (94, 73, 52), (0, 84, 211), (15, 196, 241), (156, 188, 26), (185, 128, 41),
    (173, 68, 142), (141, 140, 127), (43, 57, 192), (34, 126, 230), (96, 174, 39),
    (241, 240, 236), (166, 165, 149), (199, 195, 189), (80, 62, 44), (133, 160, 22),
]


def get_color(idx):
    return color_set[idx]


if __name__ == '__main__':

    num = len(color_set)

    a = 50
    sq1 = np.zeros((a, a*num, 3), dtype=np.uint8)
    sq2 = np.zeros((a, a*num, 3), dtype=np.uint8)

    for i, color in enumerate(color_set):
        cv2.rectangle(sq1, (a*i, 0), (a*i+a-1, a-1), color=color, thickness=-1)
        cv2.putText(sq1, str(i), (a * i, a//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.imshow('1', sq1)
    cv2.waitKey(0)

    # for i, idx in enumerate([10, 3, 4, 8, 12, 16, 7, 2, 0, 9, 13, 19, 11, 6, 5, 14, 18, 15, 17, 1]):
    #     color = get_color(idx)
    #     cv2.rectangle(sq2, (a * i, 0), (a * i + a - 1, a - 1), color=color, thickness=-1)
    #     cv2.putText(sq2, str(i), (a * i, a // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    #     print(color, end=', ')


    # cv2.imshow('2', sq2)
    # cv2.waitKey(0)



