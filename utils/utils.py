from __future__ import division

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import math
import time
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch
import numpy as np
from shapely.geometry import Polygon

import sys
sys.path.append('../')

# import data_process.config as cnf
import data_process.kitti_bev_utils as bev_utils

def convert_format(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / (box.union(b).area + 1e-12) for b in boxes]

    return np.array(iou, dtype=np.float32)

def to_cpu(tensor):
    return tensor.detach().cpu()

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :6]
        pred_scores = output[:, 6]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                #iou, box_index = rotated_bbox_iou(pred_box.unsqueeze(0), target_boxes, 1.0, False).squeeze().max(0)
                ious = rotated_bbox_iou_polygon(pred_box, target_boxes)
                iou, box_index = torch.from_numpy(ious).max(0)

                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def rotated_box_wh_iou_polygon(anchor, wh, imre):
    w1, h1, im1, re1 = anchor[0], anchor[1], anchor[2], anchor[3]

    wh = wh.t()
    imre = imre.t()
    w2, h2, im2, re2 = wh[0], wh[1], imre[0], imre[1]

    anchor_box = torch.cuda.FloatTensor([100, 100, w1, h1, im1, re1]).view(-1, 6)    
    target_boxes = torch.cuda.FloatTensor(w2.shape[0], 6).fill_(100)

    target_boxes[:, 2] = w2
    target_boxes[:, 3] = h2
    target_boxes[:, 4] = im2
    target_boxes[:, 5] = re2

    ious = rotated_bbox_iou_polygon(anchor_box[0], target_boxes)

    return torch.from_numpy(ious)

def rotated_box_11_iou_polygon(box1, box2, nG):

    box1_new = torch.cuda.FloatTensor(box1.shape[0], 6).fill_(0)
    box2_new = torch.cuda.FloatTensor(box2.shape[0], 6).fill_(0)

    box1_new[:, :4] = box1[:, :4]
    box1_new[:, 4:] = box1[:, 4:]

    box2_new[:, :4] = box2[:, :4] * nG
    box2_new[:, 4:] = box2[:, 4:]

    ious = []
    for i in range(box1_new.shape[0]):
        bbox1 = box1_new[i]
        bbox2 = box2_new[i].view(-1, 6)

        iou = rotated_bbox_iou_polygon(bbox1, bbox2).squeeze()
        ious.append(iou)

    ious = np.array(ious)

    return torch.from_numpy(ious)

def rotated_bbox_iou_polygon(box1, box2):
    box1 = to_cpu(box1).numpy()
    box2 = to_cpu(box2).numpy()

    x,y,w,l,im,re = box1
    angle = np.arctan2(im, re)
    bbox1 = np.array(bev_utils.get_corners(x, y, w, l, angle)).reshape(-1,4,2)
    bbox1 = convert_format(bbox1)

    bbox2 = []
    for i in range(box2.shape[0]):
        x,y,w,l,im,re = box2[i,:]
        angle = np.arctan2(im, re)
        bev_corners = bev_utils.get_corners(x, y, w, l, angle)
        bbox2.append(bev_corners)
    bbox2 = convert_format(np.array(bbox2))

    return compute_iou(bbox1[0], bbox2)

def non_max_suppression_rotated_bbox(prediction, conf_thres=0.95, nms_thres=0.4):
    """ 
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x, y, w, l, im, re, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 6] >= conf_thres]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 6] * image_pred[:, 7:].max(1)[0]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 7:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :7].float(), class_confs.float(), class_preds.float()), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            #large_overlap = rotated_bbox_iou(detections[0, :6].unsqueeze(0), detections[:, :6], 1.0, False) > nms_thres # not working
            large_overlap = rotated_bbox_iou_polygon(detections[0, :6], detections[:, :6]) > nms_thres
            # large_overlap = torch.from_numpy(large_overlap.astype('uint8'))
            large_overlap = torch.from_numpy(large_overlap)
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 6:7]
            # Merge overlapping bboxes by order of confidence
            detections[0, :6] = (weights * detections[invalid, :6]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output
