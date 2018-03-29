# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 08:49:13 2018

@author: RockyZhou
"""
import numpy as np
import utils
import random
import scipy.misc
import os
import threading
import skimage
import skimage.transform

import logging

from keras.utils import Sequence

############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)
############################################################
#  DataGenerator_Class
############################################################
class Iterator_MaskRCNN(Sequence):
    """Base class for image data iterators.
    """

    def __init__(self, n, gen_len, batch_size, shuffle, seed):
        
        self.n        = n
        self.batch_size = batch_size
    
        self.seed            = seed
        self.shuffle         = shuffle #neglect
        self.batch_index     = 0
        self.total_batches_seen = 0
        
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index() #for child classes
        
        self.gen_len  = gen_len
        
        index_array = np.arange(self.n)        
        #if the iterations for one epoch is larger than n//batch_size
        epoch_len = gen_len * batch_size
        if epoch_len > n:
            print("Extending the index array...")
            n_times = epoch_len//n
            remainder = epoch_len - n_times*n
            repeat_index_array = np.repeat(index_array, n_times)
            remainder_index_array = np.random.randint(self.n, size=remainder)
            
            index_array = np.hstack((repeat_index_array, remainder_index_array))
        
        print('Current index size: '+ str(index_array.size))
        assert(index_array.size//batch_size >= gen_len)
        
        self.orig_index_array = index_array
        
        self.train_n   = np.size(self.orig_index_array)
        self.train_len = np.size(self.orig_index_array)//self.batch_size
        
    def _set_index_array(self):
        
        self.index_array = np.copy(self.orig_index_array)
        
        if self.shuffle:
            self.index_array = np.random.permutation(self.orig_index_array)


    def __getitem__(self, idx): #in fit_generator this function is called
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]

        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return self.train_len
        #return (self.train_len + self.augmen_batch_size - 1) // self.augmen_batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            
            #print(self.batch_index)
            if self.batch_index == 0:
                self._set_index_array()
                
            current_index = (self.batch_index * self.batch_size) % self.train_n
            
            #if self.train_n > current_index + self.batch_size:
            if self.train_len > self.batch_index:
                self.batch_index += 1
            else:
                self.batch_index = 0
                
            self.total_batches_seen += 1
            
            yield self.index_array[current_index:
                                   current_index + self.batch_size]
            

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        """
        raise NotImplementedError
        
class CustomDatasetIterator_MaskRCNN(Iterator_MaskRCNN):

    def __init__(self, dataset,
                 config,
                 mode='val',
                 random_rois=0,
                 detection_targets=False,
                 batch_size=1, shuffle=True, augment=True, seed=None,
                 save_to_dir=None,
                 save_prefix='', save_format='png'):#(dataset, config, shuffle=True, augment=True, random_rois=0,
                   #batch_size=1, detection_targets=False)
        
        self.dataset     = dataset
        self.image_ids   = np.copy(dataset.image_ids)
        self.num_samples = np.size(self.image_ids)
        
        print("Found %d images." % (self.num_samples))
        
        #Preload
        image_id = self.image_ids[0]
        image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_kp_vs, gt_kp_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              use_mini_mask=config.USE_MINI_MASK)
        self.image_meta = image_meta
        self.image      = image
        print("Image Meta shape:")
        print(image_meta.shape)
        
        self.config  = config
        if mode == 'val':
            self.generator_len = self.num_samples // batch_size
        else:
            self.generator_len = config.STEPS_PER_EPOCH
        
        self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)
        
        
        self.batch_size=batch_size
        
        self.augment = augment
        self.random_rois = random_rois
        self.detection_targets=detection_targets
        
        self.error_count = 0
        
        #
        super(CustomDatasetIterator_MaskRCNN, self).__init__(self.num_samples, self.generator_len, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        
        batch_size = self.batch_size
        anchors    = self.anchors
        dataset    = self.dataset
        config     = self.config
        
        image_meta = self.image_meta
        image      = self.image
        
        assert(batch_size == len(index_array))
        
        batch_image_meta = np.zeros(
            (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
        batch_rpn_match = np.zeros(
            [batch_size, anchors.shape[0], 1], dtype=np.int32)
        batch_rpn_bbox = np.zeros(
            [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=np.float32)
        batch_images = np.zeros(
            (batch_size,) + image.shape, dtype=np.float32)
        batch_gt_class_ids = np.zeros(
            (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
        batch_gt_boxes = np.zeros(
            (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
        
        if config.USE_MINI_MASK:
            batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                       config.MAX_GT_INSTANCES))
        else:
            batch_gt_masks = np.zeros(
                (batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
        
        ##
        #kps
        batch_gt_kp_vs = np.zeros(
            (batch_size, config.MAX_GT_INSTANCES, 17), dtype=np.int32)
        batch_gt_kp_masks = np.zeros((batch_size, config.KP_MASK_SHAPE[0], config.KP_MASK_SHAPE[1], 17, config.MAX_GT_INSTANCES))
#        if config.USE_MINI_MASK:
#            batch_gt_kp_masks = np.zeros((batch_size, config.KP_MASK_SHAPE[0], config.KP_MASK_SHAPE[1], 17, config.MAX_GT_INSTANCES))
#        else:
#            batch_gt_kp_masks = np.zeros(
#                (batch_size, image.shape[0], image.shape[1], 17, config.MAX_GT_INSTANCES))
        
#        #not checked
#        if self.random_rois:
#            batch_rpn_rois = np.zeros(
#                (batch_size, self.random_rois, 4), dtype=np.int32)
#            if self.detection_targets:
#                batch_rois = np.zeros(
#                    (batch_size,) + (config.TRAIN_ROIS_PER_IMAGE, 4), dtype=np.int32)
#                batch_mrcnn_class_ids = np.zeros(
#                    (batch_size,) + (config.TRAIN_ROIS_PER_IMAGE,), dtype=np.int32)
#                batch_mrcnn_bbox = np.zeros(
#                    (batch_size,) + (config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 4), dtype=np.float32)
#                batch_mrcnn_mask = np.zeros(
#                    (batch_size,) + (config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES), dtype=np.float32)
        
        
        for i,j in enumerate(index_array):
            
            try:
                # Get GT bounding boxes and masks for image.
                image_id = self.image_ids[j]
                image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_kp_vs, gt_kp_masks = \
                    load_image_gt(dataset, config, image_id, augment=self.augment,
                                  use_mini_mask=config.USE_MINI_MASK)
    
                # Skip images that have no instances. This can happen in cases
                # where we train on a subset of classes and the image doesn't
                # have any of the classes we care about.
                if not np.any(gt_class_ids > 0):
                    continue
                
                # RPN Targets
                rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                        gt_class_ids, gt_boxes, config)
    
    #            # Mask R-CNN Targets
    #            if self.random_rois:
    #                rpn_rois = generate_random_rois(
    #                    image.shape, self.random_rois, gt_class_ids, gt_boxes)
    #                if self.detection_targets:
    #                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
    #                        build_detection_targets(
    #                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)
                            
                            
                # If more instances than fits in the array, sub-sample from them.
                if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                    ids = np.random.choice(
                        np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    gt_masks = gt_masks[:, :, ids]
                    gt_kp_vs = gt_kp_vs[ids]
                    gt_kp_masks = gt_kp_masks[:,:,:,ids]
                    
                # Add to batch
                batch_images[i] = mold_image(image.astype(np.float32), config) #subtract the mean
                batch_image_meta[i] = image_meta
                batch_rpn_match[i] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[i] = rpn_bbox
                
                batch_gt_class_ids[i, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[i, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[i, :, :, :gt_masks.shape[-1]] = gt_masks
    
                batch_gt_kp_vs[i, :gt_kp_vs.shape[0]] = gt_kp_vs
                batch_gt_kp_masks[i, :, :, :,:gt_kp_masks.shape[-1]] = gt_kp_masks
    
    #            if self.random_rois:
    #                batch_rpn_rois[i] = rpn_rois
    #                if self.detection_targets:
    #                    batch_rois[i] = rois
    #                    batch_mrcnn_class_ids[i] = mrcnn_class_ids
    #                    batch_mrcnn_bbox[i] = mrcnn_bbox
    #                    batch_mrcnn_mask[i] = mrcnn_mask
            except (GeneratorExit, KeyboardInterrupt):
                raise
            except:
                # Log it and skip the image
                logging.exception("Error processing image {}".format(
                    dataset.image_info[image_id]))
                self.error_count += 1
                if self.error_count > 5:
                    raise
                        
        inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, 
                  batch_gt_kp_vs, batch_gt_kp_masks]
        outputs = []

#        if self.random_rois:
#            inputs.extend([batch_rpn_rois])
#            if self.detection_targets:
#                inputs.extend([batch_rois])
#                # Keras requires that output and targets have the same number of dimensions
#                batch_mrcnn_class_ids = np.expand_dims(
#                    batch_mrcnn_class_ids, -1)
#                outputs.extend(
#                    [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])
                
#        # optionally save augmented images to disk for debugging purposes
#        if self.save_to_dir:
#            for i, j in enumerate(index_array):
#                fname = self.vessel_filenames[j]
#                img=array_to_img(batch_x[i]+128.,self.data_format,scale=False)               
#                gdt=array_to_img(np.reshape(batch_y[i,:,:,0], self.target_size + (1,)),self.data_format,scale=False)
#                
#                save_to_disk(img, gdt, prefix=self.save_prefix, fname=fname, save_to_dir=self.save_to_dir, save_format=self.save_format)
        

        return inputs, outputs

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    
############################################################
#  DataGenerator
############################################################
def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0,
                   batch_size=1, detection_targets=False):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_kp_vs, gt_kp_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              use_mini_mask=config.USE_MINI_MASK)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                
                if config.USE_MINI_MASK:
                    batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                               config.MAX_GT_INSTANCES))
                else:
                    batch_gt_masks = np.zeros(
                        (batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
                
                ##
                #kps
                batch_gt_kp_vs = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 17), dtype=np.int32)
                if config.USE_MINI_MASK:
                    batch_gt_kp_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], 17, config.MAX_GT_INSTANCES))
                else:
                    batch_gt_kp_masks = np.zeros(
                        (batch_size, image.shape[0], image.shape[1], 17, config.MAX_GT_INSTANCES))
                
                #not checked
                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]
                gt_kp_vs = gt_kp_vs[ids]
                gt_kp_masks = gt_kp_masks[:,:,:,ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config) #subtract the mean
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

            batch_gt_kp_vs[b, :gt_kp_vs.shape[0]] = gt_kp_vs
            batch_gt_kp_masks[b, :, :, :,:gt_kp_masks.shape[-1]] = gt_kp_masks

            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, 
                          batch_gt_kp_vs, batch_gt_kp_masks]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend(
                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise
                
def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    #kps (17*3, instance_num)
    mask, class_ids, kps = dataset.load_mask(image_id)
    #print(kps.shape)
    shape = image.shape
    if augment:
        image, window, scale, padding = aug_resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
    else:
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
    
    mask = resize_mask(mask, scale, padding)
    kp_vs, kp_masks = resize_kp_mask(kps, image.shape[:2], scale , padding)

    # Random horizontal flips.
    if augment:
        flip = random.randint(0, 1)
        #print('flip'+ str(flip))
        #flip=1
        if flip: 
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            #This is important to avoid the mirror person problem
            kp_vs, kp_masks = flipkeypoints(kp_vs,kp_masks)
            #kp_masks = np.fliplr(kp_masks)

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1 # classes defined in this dataset

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        
    kp_vs, kp_masks = minimize_kp_mask(bbox, kp_vs, kp_masks, config.KP_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)

    return image, image_meta, class_ids, bbox, mask, kp_vs, kp_masks

def flipkeypoints(kp_vs,kp_masks):
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    
    #final_kp_vs:     (instance_num, 17)
    #final_kp_masks:  (H,W, 17, instance_num)
    final_kp_vs   = np.zeros(kp_vs.shape, dtype=np.int32)
    final_kp_masks = np.zeros(kp_masks.shape)
    
    final_kp_vs[:,0]=kp_vs[:,0] 
    final_kp_vs[:,1::2]=kp_vs[:,2::2]
    final_kp_vs[:,2::2]=kp_vs[:,1::2]
    
    flipped_masks = np.fliplr(kp_masks)
    final_kp_masks[:,:,0,:]=flipped_masks[:,:,0,:]
    final_kp_masks[:,:,1::2,:]=flipped_masks[:,:,2::2,:]
    final_kp_masks[:,:,2::2,:]=flipped_masks[:,:,1::2,:] 
#    for n in range(kp_vs.shape[0]):
#        vs = kp_vs[n]
#        final_kp_vs[n,0]=vs[0] 
#        final_kp_vs[n,1::2]=vs[2::2]
#        final_kp_vs[n,2::2]=vs[1::2]
#        
#        
#        flipped_mask = np.fliplr(kp_masks[:,:,:,n])
#        final_kp_masks[:,:,0,n]=flipped_mask[:,:,0]
#        final_kp_masks[:,:,1::2,n]=flipped_mask[:,:,2::2]
#        final_kp_masks[:,:,2::2,n]=flipped_mask[:,:,1::2]   

    return final_kp_vs, final_kp_masks

def aug_resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        min_dim = np.random.randint(640, 800)
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        #print("Scale" + str(scale))
        image = skimage.transform.resize(image, (round(h*scale), round(w*scale)), mode='constant', preserve_range=True)
#        image = scipy.misc.imresize(
#            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding

def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = skimage.transform.resize(mask, (round(h*scale), round(w*scale)), order=0, mode='constant', preserve_range=True)
    #mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask
   
def resize_kp_mask(kps, new_size, scale, padding):
    #Return
    #final_kp_vs:     (instance_num, 17)
    #final_kp_masks:  (H,W, 17, instance_num)
    
    #kps (17*3, instance_num)
    assert(kps.shape[0]==51)    
    final_kp_vs   = np.zeros((kps.shape[1], 17), dtype=np.int32)
    final_kp_masks = np.zeros(new_size + (17,) + (kps.shape[1],))

    for n in range(kps.shape[1]):
        kp = kps[:,n]
        final_kp_vs[n] = kp[2::3]
        x = kp[0::3]
        y = kp[1::3]
        
        y = np.floor(y * scale)
        x = np.floor(x * scale)
        #y = int(y + padding[0][0])  # axis = y
        #x = int(x + padding[1][0])  # axis = x
        y = np.array(y + padding[0][0], dtype=np.int32)
        x = np.array(x + padding[1][0], dtype=np.int32)
        
        final_kp_masks[y, x, np.arange(17), n] = 1        

    return final_kp_vs, final_kp_masks
    
def minimize_kp_mask(bbox, kp_vs, kp_mask, mini_shape):
    
    #bbox [instance_num, 4]
    #kp_vs  [instance, 17]
    #kp_mask [H,W, 17, instance]
    mini_kp_vs   = kp_vs
    mini_kp_mask = np.zeros( mini_shape + (17,) + (kp_mask.shape[-1],), dtype=bool)
        
    for n in range(bbox.shape[0]):
        y1, x1, y2, x2 = bbox[n]
        m = kp_mask[:,:,:,n]
        m = m[y1:y2, x1:x2, :] #(ch,cw,17)

        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")

        scale = np.asarray(mini_shape) / m.shape[:2]

        if m.sum() == 0:
            mini_kp_vs[n] = 0 #all not exist
        else:
            y,x,z = np.where(m == 1)
            y = np.array(y*scale[0], dtype=np.int32)
            x = np.array(x*scale[1], dtype=np.int32)
            
            mini_kp_mask[y, x, z, n] = 1
            #TODO: make sure mini_kp_vs at corresponding position has the right label

#        if m.size == 0:
#            raise Exception("Invalid bounding box with area of zero")
#            
#        for i in range(17):
#            m_single = m[:,:,i]
#
#            if m_single.sum() == 0:
#                m_single[0, 0] = 1
#                mini_kp_vs[n, i] = -kp_vs[n,i]
#            else:
#                cord = np.where(m_single == int(m_single.max()))
#                new_cord = np.array([cord[0] * scale[0], cord[1] * scale[1]], dtype=np.int32).reshape(2,)
#                mini_kp_mask[new_cord[0], new_cord[1], i, n] = 1

    return mini_kp_vs, mini_kp_mask
    
############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta
    
############################################################
#  Build Targets
############################################################  
def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox
    
def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL
    

############################################################
#  To Debug
############################################################    
def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois
    
    
def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Grund truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinments.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
#     bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indicies of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinments
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks.
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(scipy.misc.imresize(class_mask.astype(float), (gt_h, gt_w),
                                             interp='nearest') / 255.0).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = scipy.misc.imresize(
            m.astype(float), config.MASK_SHAPE, interp='nearest') / 255.0
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks