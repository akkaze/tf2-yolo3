import tensorflow as tf
from tensorflow.keras.layers import Lambda
from .models import yolo_boxes, yolo_nms


def make_eval_model_from_trained_model(model, anchors, masks, num_classes=10, tiny=True):
    if tiny:
        output_0, output_1 = model.outputs
        boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], num_classes, False), name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], num_classes, False), name='yolo_boxes_1')(output_1)
        outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, num_classes),
                         name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
        model = tf.keras.Model(model.inputs, outputs, name='yolov3_tiny')
    else:
        output_0, output_1, output_2 = model.outputs
        boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], num_classes, False), name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], num_classes, False), name='yolo_boxes_1')(output_1)
        boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], num_classes, False), name='yolo_boxes_1')(output_2)
        outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, num_classes),
                         name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))
        model = tf.keras.Model(model.inputs, outputs, name='yolov3')
    return model