from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import visdom
import numpy as np
import cv2
import _init_paths
from datasets.factory import get_imdb
import os.path as osp
from fast_rcnn.config import cfg
import os



import pickle

import os
import os.path as osp

import numpy as np
from fast_rcnn.config import cfg
import time



imdb = get_imdb('voc_2007_trainval')
# imdb.selective_search_roidb()


cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
data_name = "voc_2007_trainval"
roidbd_name = ["_selective_search_roidb.pkl", '_gt_roidb.pkl']
cache_file_ss = os.path.join(cache_path, data_name + roidbd_name[0])
cache_file_gt = os.path.join(cache_path, data_name + roidbd_name[1])


# with open(cache_file_ss, 'rb') as fid:
#     try:
#         ss_roidb = pickle.load(fid)
#     except:
#         ss_roidb = pickle.load(fid, encoding='bytes')

with open(cache_file_gt, 'rb') as fid:
    try:
        gt_roidb = pickle.load(fid)
    except:
        gt_roidb = pickle.load(fid, encoding='bytes')

# print(ss_roidb[0].keys())
print(gt_roidb[0].keys())
print(type(list(gt_roidb[0]['gt_classes'])))
print(gt_roidb[0]['gt_classes'].shape)
print(list(np.unique(gt_roidb[0]['gt_classes'])))
time.sleep(15000)
'''
    show the selective search results
'''
tmp = ss_roidb[2020]['boxscores'].ravel()
index = np.argsort(tmp)[::-1]
bbox = ss_roidb[2020]['boxes'][index]
scores = tmp[index]



'''
    show the ground-truth results
'''
gt_bbox = gt_roidb[2020]['boxes']
gt_class_name = gt_roidb[2020]['gt_classes']
print(gt_bbox.shape)


img = cv2.imread(imdb.image_path_at(2020))
vis = visdom.Visdom(server='http://localhost',port='8097')

def vis_gtdetections(im, dets):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
    return im


def vis_detections1(im, dets, scores, thresh=0.010):
    """Visual debugging of detections."""
    count = 1
    color = (255, 0, 0)
    thickness = 2

    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = scores[i]
        if count>10:
            break
        if score > thresh:
            count+=1
            cv2.rectangle(im, tuple(bbox[0:2]), tuple(bbox[2:4]), color, thickness)
    return im

print(gt_bbox.shape)
gt_img = vis_gtdetections(img, gt_bbox)
gt_img = gt_img.transpose(2,0,1)
vis.image(gt_img)


new_img = vis_detections1(img, bbox, scores)
new_img = new_img.transpose(2,0,1)
vis.image(new_img)
