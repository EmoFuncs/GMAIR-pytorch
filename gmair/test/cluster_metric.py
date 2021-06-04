import os
import numpy as np
import cv2
import torch

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score

from gmair.utils.bbox.bbox import bbox_overlaps
from gmair.config import config as cfg

from ipdb import set_trace

def get_bbox_labels(z_where, z_cls, obj_prob, ground_truth_bbox, conf_thresh = 0.5, iou_thresh = 0.5):
    image_size = cfg.input_image_shape[-1]
    batch_size = z_where.size(0)

    z_where = z_where.view(batch_size, -1, 4).detach().cpu().numpy()
    z_cls = z_cls.view(batch_size, -1, cfg.num_classes).detach().cpu().numpy()
    obj_prob = obj_prob.view(batch_size, -1, 1).detach().cpu().numpy()

    z_where[..., :2] -= z_where[..., 2:]/2
    z_where *= image_size
    
    z_pred = np.concatenate((z_where, obj_prob), axis = 2)
    
    ground_truth_bbox = ground_truth_bbox.detach().cpu().numpy()

    true_labels = np.zeros(0, dtype=np.int64)
    pred_labels = np.zeros(0, dtype=np.int64)
    
    for i in range(batch_size):
        pred_info = z_pred[i, z_pred[i,:,4]>=conf_thresh, :].astype('float64')
        pred_cls = z_cls[i, z_pred[i,:,4]>=conf_thresh, :]
        pred_cls = np.argmax(pred_cls, axis=1)
        
        gt_boxes = ground_truth_bbox[i, ground_truth_bbox[i,:,0]>=0, :].astype('float64')
        
    
        _pred = pred_info.copy()
        _gt = gt_boxes.copy()
        true_label = np.zeros(_pred.shape[0], dtype=np.int64)
    
        _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
        _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

        overlaps = bbox_overlaps(_pred[:, :4], _gt[:, :4])

        if _gt.shape[0] != 0:
            for h in range(_pred.shape[0]):
                gt_overlap = overlaps[h]
                max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
                if max_overlap >= iou_thresh:
                    true_label[h] = _gt[max_idx, 4]
    
        true_labels = np.concatenate((true_labels, true_label), axis = 0)
        pred_labels = np.concatenate((pred_labels, pred_cls), axis = 0)
   
    pred_labels = pred_labels[true_labels > 0]
    true_labels = true_labels[true_labels > 0] - 1
    return pred_labels, true_labels
    
def cluster_acc(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    D = int(D)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], int(Y[i])] += 1
    # print(w)
    col = np.argmax(w, axis = 1)
    # print(col)
    # row, col = linear_sum_assignment(w.max()-w)
    return sum([w[i,col[i]] for i in range(D)]) * 1.0/Y_pred.size

def nmi(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    return normalized_mutual_info_score(Y_pred, Y, average_method='arithmetic')
    
def test_cluster(z_where, z_cls, obj_prob, ground_truth_bbox):
    y_pred, y_gt = get_bbox_labels(z_where, z_cls, obj_prob, ground_truth_bbox)
    
    if y_pred.size == 0:
        return 0, 0
    
    acc = cluster_acc(y_pred, y_gt)
    nmi_value = nmi(y_pred, y_gt)
    return acc, nmi_value
