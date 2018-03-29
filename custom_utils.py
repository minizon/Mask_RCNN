#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:07:42 2018

@author: pami4
"""
from pycocotools.coco import COCO
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc
from pycocotools import mask as maskUtils

def showKPs(image, roi, kp_vs, kp_masks, instance_mask=None):
    #kp_masks (H, W, 17)
    
    y1, x1, y2, x2 = roi.astype(np.int32)
    
    crop_image = image[y1:y2, x1:x2]
    
    resize_image = scipy.misc.imresize(crop_image, kp_masks.shape[:2])
    
    plt.imshow(resize_image)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    #
    #if not instance_mask:
    m = (instance_mask>0.5).astype(np.uint8)
    img = np.ones( (m.shape[0], m.shape[1], 3) )
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        img[:,:,i] = color_mask[i]
    ax.imshow(np.dstack( (img, m*0.25) ))
    
    #        
    coco = COCO('../annotations/person_keypoints_train2014.json')
    
    #1 is the person class id
    catInfo = coco.loadCats(coco.getCatIds())[0]
    #skeleton
    sks = np.array(catInfo['skeleton'])-1
    name_keypoints = catInfo['keypoints'] #list
    
    kp_m = np.transpose(kp_masks, axes=(2, 0, 1))
    kp_f = np.reshape(kp_m, (17, -1))
    index_array = np.argmax(kp_f, axis=1)
    
    y,x = np.unravel_index(index_array, kp_masks.shape[:2])
    v = kp_vs#np.array(kp_vs)
    for i in np.arange(v.size):
        if kp_masks[y[i],x[i],i]>0:
            pass
        else:
            v[i]=-v[i]
            
    c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
    for sk in sks:
        if np.all(v[sk]>0):
            plt.plot(x[sk],y[sk], linewidth=3, color=c)
    plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
    plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
    for i,name in enumerate(name_keypoints):
        if v[i]>0 and kp_masks[y[i],x[i],i]>0:
#            if v[i]>0:
#                plt.plot(x[i], y[i],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
#            if v[i]>1:
#                plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
            plt.text(x[i],y[i], name)
            
    
    plt.show()
    