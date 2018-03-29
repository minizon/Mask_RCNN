# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 07:50:39 2018

@author: RockyZhou
"""
import tensorflow as tf
import keras.layers as KL
import keras.engine as KE
import keras.backend as K
from keras.initializers import Constant
from net_backbone import BatchNorm
import numpy as np

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)

class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, image_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(
            self.image_shape[0] * self.image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1], )
        

class UpSamplingLayer(KE.Layer):
    
    def __init__(self, factor, **kwargs):
        self.factor = factor
        super(UpSamplingLayer, self).__init__(**kwargs)
    

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
#        self.kernel = self.add_weight(name='kernel', 
#                                      shape=(input_shape[1], self.output_dim),
#                                      initializer='uniform',
#                                      trainable=True)
        self.new_height = input_shape[1]*self.factor
        self.new_width = input_shape[2]*self.factor
        super(UpSamplingLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return tf.image.resize_images(x, [self.new_height, self.new_width])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.new_height, self.new_width, 17)

#https://kivantium.net/keras-bilinear    
def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    
    filter_size = 2 * factor - factor % 2
    
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in range(number_of_classes):
        
        weights[:, :, i, i] = upsample_kernel
    
    return weights

def build_fpn_kp_mask_graph(rois, feature_maps,
                         image_shape, pool_size, num_kp_classes):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_kp_classes: number of classes, which determines the depth of the results

    Returns: Masks [batch, roi_count, height, width, num_kp_classes]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_kp_mask")([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn4')(x)
    x = KL.Activation('relu')(x)
    
    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv5")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn5')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv6")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn6')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv7")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn7')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv8")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn8')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(17, (2, 2), strides=2), name="mrcnn_kp_mask_deconv")(x)
    
    x = KL.TimeDistributed(KL.Conv2DTranspose(17, (4, 4), strides=2, padding='same',
                                              kernel_initializer=Constant(bilinear_upsample_weights(2, 17)) ),
                           name="kp_mask_bilinear_up")(x) #not trainable
    x = KL.Activation('sigmoid')(x)
     
#    x = KL.TimeDistributed(KL.Conv2D(17, (1, 1), strides=1, activation="linear"),
#                           name="mrcnn_kp_mask")(x)
    
#    x = KL.TimeDistributed(UpSamplingLayer(2),
#                           name="mrcnn_kp_mask_bilinear_up")(x)
    return x

def mrcnn_kp_mask_loss_graph_v2(target_masks, target_class_ids, pred_masks):
    """
    target_masks: [batch, num_rois, height, width, 17].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, num_rois, height, width, 17] float32 tensor
                with values from 0 to 1.
    """
    #**************************************************************
    #!!!!!!!!!!!!!!
    #The same problem:  Ran out of GPU memory when allocating 0 bytes
    #*************************************************************
    
    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    target_class_ids = K.reshape(target_class_ids, (-1,))

    positive_ix = tf.where(target_class_ids > 0)[:, 0]

    # Predicted masks and target masks are reshaped in [N, height*width, 14 (body parts)]
    # mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, 28 * 28, 17))
    pred_masks = K.reshape(pred_masks, (-1, 28 * 28, 17))

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather(pred_masks, positive_ix)

    # compute the loss function in second dimmention (28*28), result = [N, 17]
    loss = K.switch(tf.size(y_true) > 0,
                    tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1),
                    tf.constant(0.0))

    loss = K.mean(loss)
    
    loss = K.reshape(loss, [1, 1])
    return loss
    
    
def mrcnn_kp_mask_loss_graph(target_masks, target_kp_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width, kp_num].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_kp_class_ids: [batch, num_rois, kp_num]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, kp_num] float32 tensor
                with values from 0 to 1.
    """
    with tf.device('/cpu:0'):
        
        # Permute predicted masks to [batch, proposals, kp_num, height, width]
        #print(tf.shape(target_masks))
        target_masks = tf.transpose(target_masks, [0, 1, 4, 2, 3])
        mask_shape = tf.shape(target_masks)
        target_masks = K.reshape(target_masks, (-1, mask_shape[-2]*mask_shape[-1]))
        
        pred_masks = tf.transpose(pred_masks, [0, 1, 4, 2, 3])
        pred_shape = tf.shape(pred_masks)
        pred_masks = K.reshape(pred_masks, (-1, pred_shape[-2]*pred_shape[-1]))
        
        target_kp_class_ids = K.reshape(target_kp_class_ids, (-1,))
        
        positive_ix = tf.where(target_kp_class_ids > 1)[:, 0] #only visible
        
        #positive_class_ids = tf.cast(
        #    tf.gather(target_class_ids, positive_ix), tf.int64)
        #indices = tf.stack([positive_ix, positive_class_ids], axis=1)
    
        # Gather the masks (predicted and true) that contribute to loss
        y_true = tf.gather(target_masks, positive_ix) #(batch*proposals*kp_num, height*width)
        
        y_pred = tf.gather(pred_masks, positive_ix) #(batch*proposals*kp_num, height*width)
    #    y_pred = tf.gather_nd(pred_masks, indices)
        
        # compute the loss function in second dimmention (56*56)
        loss = K.switch(tf.size(y_true) > 0,
                        tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true),
                        tf.constant(0.0))
    
        # loss = tf.gather(K.reshape(loss, (-1,)), target_mask_class)
        loss = K.mean(loss)
        
        loss = K.reshape(loss, [1, 1])
    return loss
    
def mrcnn_kp_vs_loss_graph(target_mask_class, target_class_ids, pred_class):
    """Loss for Mask class R-CNN whether key points are in picture.

        target_mask_class: [batch, num_rois, 17(number of keypoints)]
        pred_class: [batch, num_rois, 17, 3]
        target_class_ids: [batch, num_rois]. Integer class IDs.
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_mask_class = tf.cast(target_mask_class, tf.int64)
    
    target_class_ids = K.reshape(target_class_ids, (-1,))
    
    pred_class = K.reshape(pred_class, (-1, 17, 3)) #K.int_shape(pred_class)[3]))
    target_mask_class = tf.cast(K.reshape(target_mask_class, (-1, 17)), tf.int64)

    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]

    # Gather the positive classes (predicted and true) that contribute to loss
    target_class = tf.gather(target_mask_class, positive_roi_ix)
    pred_class = tf.gather(pred_class, positive_roi_ix)

#    # Loss
#    loss = K.switch(tf.size(target_class) > 0,
#                    lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class, logits=pred_class),
#                    lambda: tf.constant(0.0))
#    # Computer loss mean. Use only predictions that contribute
#    # to the loss to get a correct mean.
#    loss = tf.reduce_mean(loss)
    
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class, logits=pred_class)
    loss = K.sparse_categorical_crossentropy(target=target_class,
                                             output=pred_class,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    
    return loss


def build_fpn_shared_kp_mask_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_kp_classes: number of classes, which determines the depth of the results

    Returns: Masks [batch, roi_count, height, width, num_classes]
    """
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_mask")([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn4')(x)
    shared = KL.Activation('relu')(x)
    
    
    ##mask branch
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(shared)
    mrcnn_mask_probs = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), 
                                          name="mrcnn_mask")(x)
    
    ##kp branch
    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv5")(shared)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn5')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv6")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn6')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv7")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn7')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_kp_mask_conv8")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_kp_mask_bn8')(x)
    x = KL.Activation('relu')(x)

#    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2), name="mrcnn_kp_mask_deconv")(x)
#    
#    mrcnn_kp_mask = KL.TimeDistributed(KL.Conv2D(17, (1, 1), strides=1, activation="tanh"), 
#                                          name="mrcnn_kp_mask")(x)
    mrcnn_kp_mask = KL.TimeDistributed(KL.Conv2DTranspose(17, (2, 2), strides=2, activation="tanh"), name="mrcnn_kp_mask_deconv")(x)
#    mrcnn_kp_mask = KL.TimeDistributed(KL.Conv2DTranspose(17, (4, 4), strides=2, padding='same',
#                                              kernel_initializer=Constant(bilinear_upsample_weights(2, 17)) ),
#                           name="kp_mask_bilinear_up")(mrcnn_kp_mask) #not trainable
    
    #Scaling before cross entropy with logits
    mrcnn_kp_mask = KL.Lambda(lambda x: x * 10, name="output_mrcnn_kp_mask")(mrcnn_kp_mask)
    

#    mrcnn_kp_mask = KL.Activation('sigmoid')(x)
    
    return mrcnn_mask_probs, mrcnn_kp_mask