#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:22:59 2018

@author: pami4
"""

#CUDA_VISIBLE_DEVICES=0 python
from pycocotools.coco import COCO
import coco
import numpy as np
from matplotlib import pyplot as plt
import visualize

import custom_utils

config = coco.CocoConfig()
config.GPU_COUNT = 1

import CustomDataset
data_train=CustomDataset.CocoDataset()
data_train.load_coco("..","train", year=2014)
data_train.prepare()

#import CustomDataGenerator
#data_gen=CustomDataGenerator.data_generator(data_train, config, batch_size=2, shuffle=False, augment=False)
from CustomDataGenerator import CustomDatasetIterator_MaskRCNN
data_gen = CustomDatasetIterator_MaskRCNN(data_train, config, mode="val", shuffle=False,
                                         batch_size=2, augment=True)

#plt.imshow((images[0]+config.MEAN_PIXEL).astype(np.uint8))

import model as modellib
model=modellib.MaskRCNN(mode="training", config=config, model_dir="logs")
model.load_weights("logs/coco20180327T1023/mask_rcnn_coco_0050.h5", by_name=True, skip_mismatch=True)
#model.load_weights("/home/pami4/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)

inputs, outputs= next(data_gen)

outs=model.keras_model.predict(inputs)

#out_kp_vs = outs[18]
#np.where(out_kp_vs)
#
#
#out_kp_masks=outs[19]

images = inputs[0]
rois = outs[8]
img_idx=0
visualize.draw_boxes((images[img_idx]+config.MEAN_PIXEL).astype(np.uint8), boxes=rois[img_idx][:10]*np.array([1024,1024,1024,1024]))
plt.show()

#layer=model.keras_model.get_layer(name='kp_mask_bilinear_up')
#layer.get_weights()[0].shape

kp_masks=outs[-3]
kp_vs = outs[-4]
target_masks = outs[-1]
target_class_ids=outs[-2]
pred_kp_masks=outs[10]
pred_masks = outs[6]
#target_class_ids.shape



img_idx=0
index=1
visualize.draw_boxes((images[img_idx]+config.MEAN_PIXEL).astype(np.uint8), boxes=rois[img_idx][index:index+1]*np.array([1024,1024,1024,1024]))
plt.show()

custom_utils.showKPs((images[img_idx]+config.MEAN_PIXEL).astype(np.uint8), rois[img_idx][index]*np.array([1024,1024,1024,1024]),kp_vs[img_idx][index], kp_masks[img_idx][index], target_masks[img_idx][index])

plt.imshow(np.sum(kp_masks[1][index], axis=2))
plt.show()

#custom_utils.showKPs((images[1]+config.MEAN_PIXEL).astype(np.uint8), rois[1][index]*np.array([1024,1024,1024,1024]),kp_vs[1][index], kp_masks[1][index])

#pred_kp_masks=outs[10]
#pred_masks = outs[6]
#custom_utils.showKPs((images[1]+config.MEAN_PIXEL).astype(np.uint8), rois[1][index]*np.array([1024,1024,1024,1024]),kp_vs[1][index], pred_kp_masks[1][index])
custom_utils.showKPs((images[img_idx]+config.MEAN_PIXEL).astype(np.uint8), rois[img_idx][index]*np.array([1024,1024,1024,1024]),kp_vs[img_idx][index], pred_kp_masks[img_idx][index], pred_masks[img_idx][index][:,:,1])

from imp import reload
