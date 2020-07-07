import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Lambda, TimeDistributed, Conv2DTranspose

import utils
from model.models import apply_box_deltas_graph, clip_boxes_graph
from utils import parse_image_meta_graph, log2_graph


class ProposalLayer(tf.keras.layers.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates
    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)

        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):

        scores = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])

        anchors = inputs[2]

        pre_nms_anchors = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name='top_anchors').indices

        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)

        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x))

        window = np.array([0, 0, 1, 1], dtype=np.float32)

        boxes = utils.batch_slice(
            [pre_nms_anchors, deltas],
            lambda x, y: apply_box_deltas_graph(x, y),
            self.config.IMAGES_PER_GPU, names=['refined_anchors'])

        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count, self.nms_threshold, name='rpn_non_max_suppresion'
            )

            proposals = tf.gather(boxes, indices)
            padding = tf.maximum(self.proposal_count - tf.shape(proposals))
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])

            return proposals

        proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)

        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


class PyramidROIAlign(tf.keras.layers.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]
    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = pool_shape

    def call(self, inputs):
        boxes = inputs[0]

        image_meta = inputs[1]

        feature_maps = inputs[2:]

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)

        h = y2 - y1

        w = x2 - x1

        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]

        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)

        roi_level = log2_graph(tf.sqrt(h*w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))

        roi_level = tf.squeeze(roi_level, 2)

        pooled = []

        box_to_level = []

        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            box_indices = tf.cast(ix[:, 0], tf.int32)

            box_to_level.append(ix)

            level_boxes = tf.stop_gradient(level_boxes)

            box_indices = tf.stop_gradient(box_indices)

            pooled.append(
                tf.image.crop_and_resize(
                    feature_maps[i],
                    level_boxes, box_indices, self.pool_shape, method="bilinear"))

        pooled = tf.concat(pooled, axis=0)

        box_to_level = tf.concat(box_to_level, axis=0)

        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)

        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]

        pooled = tf.gather(pooled, ix)
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)

        pooled = tf.reshape(pooled, shape)

        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.
    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN

    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                    strides=anchor_stride, name='rpn_conv_shared')(feature_map)

    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)


    rpn_class_logits = Lambda(lambda t: tf.reshape())