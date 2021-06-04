import numpy as np
import torch

from gmair.utils.bbox.bbox import bbox_overlaps

from gmair.config import config as cfg

def image_eval(pred, gt, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    cnt = 0
    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
                cnt += 1

        pred_recall[h] = cnt
    return pred_recall


def img_pr_info(thresh_num, pred_info, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    
    r_index = -1
    for t in range(thresh_num):
        thresh = 1 - (t+1)/thresh_num
        while r_index + 1 < len(pred_info) and pred_info[r_index, 4] >= thresh:
            r_index += 1
        
        pr_info[t, 0] = r_index + 1
        pr_info[t, 1] = 0 if r_index == -1 else pred_recall[r_index]
        
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_obj):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        if pr_curve[i, 1] == 0:
            _pr_curve[i, 0] = 0
            _pr_curve[i, 1] = 0
        else:
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            _pr_curve[i, 1] = pr_curve[i, 1] / count_obj
    return _pr_curve

def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    
def mAP(z_where, z_pres, ground_truth_bbox, truth_bbox_digit_count, conf_thresh = 0.5, iou_thresh = 0.5, thresh_num = 1000):
    image_size = cfg.input_image_shape[-1]
    batch_size = z_pres.size(0)
    
    assert(batch_size == z_where.size(0))
    
    z_where = z_where.view(batch_size, -1, 4).detach().cpu().numpy()
    z_pres = z_pres.view(batch_size, -1, 1).detach().cpu().numpy()

    z_where[..., :2] -= z_where[..., 2:]/2
    z_where *= image_size
    
    z_pred = np.concatenate((z_where, z_pres), axis = 2)
    ground_truth_bbox = ground_truth_bbox.detach().cpu().numpy()

    count_obj = 0
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    for i in range(batch_size):
        pred_info = z_pred[i, z_pred[i,:,4]>=conf_thresh, :].astype('float64')
        pred_info = pred_info[np.argsort(pred_info[:, 4])[::-1]]
        gt_boxes = ground_truth_bbox[i, ground_truth_bbox[i,:,0]>=0,:].astype('float64')
         
        count_obj += len(gt_boxes)

        if len(gt_boxes) == 0 or len(pred_info) == 0:
            continue
            
        pred_recall = image_eval(pred_info, gt_boxes, iou_thresh)

        _img_pr_info = img_pr_info(thresh_num, pred_info, pred_recall)

        pr_curve += _img_pr_info
    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_obj)

    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]

    ap = voc_ap(recall, propose)
    
    return ap

def object_count_accuracy(z_pres:torch.Tensor, truth_bbox_digit_count):

    batch_size = z_pres.size(0)
    z_pres = z_pres.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
    z_pres_count = z_pres.round().sum(dim = -2)

    count_accuracy = (truth_bbox_digit_count - z_pres_count).mean()
    return count_accuracy


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A B / A B = A B / (area(A) + area(B) - A B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
    
