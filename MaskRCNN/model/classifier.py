import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, TimeDistributed, Lambda, Dense, Reshape

from model.proposal import PyramidROIAlign


def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers
    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, channels]

    x = PyramidROIAlign([pool_size, pool_size], name='roi_align_classifier')([rois, image_meta] + feature_maps)

    # Two 1024 FC Layers (implemented with Conv2D for consistency)

    x = TimeDistributed(Conv2D(fc_layers_size, (pool_size, pool_size), padding='valid'), name='mrcnn_class_conv1')(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)

    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name='pool_squeeze')(x)

    mrcnn_class_logits = TimeDistributed(Dense(num_classes), name='mrcnn_class_logits')(shared)

    mrcnn_probs = TimeDistributed(Activation('softmax'), name='mrcnn_class')(mrcnn_class_logits)

    x = TimeDistributed(Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)

    s = K.int_shape(x)
    # TODO: Reshape was -> (s[1], num_classes, 4)

    mrcnn_bbox = Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox
