import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)

import tensorflow as tf
import cv2
import time
import sys
from .utils import cv2_letterbox_resize
import zipfile
from six.moves import urllib
from tqdm import tqdm
import os


@tf.function
def transform_targets_for_output(y_true, grid_y, grid_x, anchor_idxs, classes):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((N, grid_y, grid_x, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2.

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_size = tf.cast(tf.stack([grid_x, grid_y], axis=-1), tf.float32)
                grid_xy = tf.cast(box_xy * grid_size, tf.int32)
                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    y_ture_out = tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())
    return y_ture_out


def transform_targets(y_train, size, anchors, anchor_masks, classes, tiny=True):
    y_outs = []
    if tiny:
        grid_y, grid_x = size[0] // 16, size[1] // 16
    else:
        grid_y, grid_x = size[0] // 32, size[1] // 32
    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_out = transform_targets_for_output(y_train, grid_y, grid_x, anchor_idxs, classes)
        y_outs.append(y_out)
        grid_x *= 2
        grid_y *= 2

    return tuple(y_outs)


def decode_line(line, size):
    # Decode the line to tensor
    line = line.numpy().decode()
    line_parts = line.strip().split()
    imgname = line_parts[0]
    x_train = cv2.imread(imgname)
    #x_train = transform_images(x_train, size)
    x_train, amat = cv2_letterbox_resize(x_train, (size, size))
    x_train = x_train / 255.
    xmins, ymins, xmaxs, ymaxs, labels = [], [], [], [], []
    bbox_with_labels = line_parts[1:]
    for bbox_with_label in bbox_with_labels:
        bbox_with_label_parts = bbox_with_label.split(',')
        xmin = float(bbox_with_label_parts[0])
        ymin = float(bbox_with_label_parts[1])
        xmax = float(bbox_with_label_parts[2])
        ymax = float(bbox_with_label_parts[3])
        tl = np.array([xmin, ymin, 1], np.float32)
        br = np.array([xmax, ymax, 1], np.float32)
        tl = np.dot(amat, tl)
        br = np.dot(amat, br)
        xmin, ymin = tl[0], tl[1]
        xmax, ymax = br[0], br[1]
        xmins.append(xmin / size)
        ymins.append(ymin / size)
        xmaxs.append(xmax / size)
        ymaxs.append(ymax / size)
        labels.append(float(bbox_with_label_parts[4]))
    assert np.all(np.array(xmins) <= 1)
    y_train = np.stack((xmins, ymins, xmaxs, ymaxs, labels), axis=1)
    paddings = [[0, 100 - y_train.shape[0]], [0, 0]]
    y_train = np.pad(y_train, paddings, mode='constant')
    return x_train, y_train


def load_textline_dataset(file_pattern, size):
    dataset = tf.data.TextLineDataset(file_pattern)
    return dataset.map(lambda x: tf.py_function(func=decode_line, inp=[x, size], Tout=(tf.float32, tf.float32)))


def download_m2nist_if_not_exist():
    data_rootdir = os.path.expanduser('~/.m2nist')
    m2nist_zip_path = os.path.join(data_rootdir, 'm2nist.zip')
    if os.path.exists(m2nist_zip_path):
        return
    os.makedirs(data_rootdir, exist_ok=True)
    m2nist_zip_url = 'https://raw.githubusercontent.com/akkaze/datasets/master/m2nist.zip'
    while True:
        try:
            download_from_url(m2nist_zip_url, m2nist_zip_path)
            break
        except:
            time.sleep(5)
            continue
    zipf = zipfile.ZipFile(m2nist_zip_path)
    zipf.extractall(data_rootdir)
    zipf.close()


def load_m2nist_dataset(dst_size=(64, 64), val_ratio=0.2):
    download_m2nist_if_not_exist()
    data_rootdir = os.path.expanduser('~/.m2nist')
    imgs = np.load(os.path.join(data_rootdir, 'combined.npy')).astype(np.uint8)

    num_data = imgs.shape[0]
    num_train = int(num_data * (1 - val_ratio))

    def transform_target(img, line, expected_size):
        img = img.numpy()
        line = line.numpy().decode()
        expected_size = tuple(expected_size.numpy())
        img, amat = cv2_letterbox_resize(img, expected_size)
        bbox_with_labels = line.strip().split()[1:]
        xmins, xmaxs, ymins, ymaxs, labels = [], [], [], [], []
        for bbox_with_label in bbox_with_labels:
            bbox_with_label_parts = bbox_with_label.split(',')
            xmin = float(bbox_with_label_parts[0])
            ymin = float(bbox_with_label_parts[1])
            xmax = float(bbox_with_label_parts[2])
            ymax = float(bbox_with_label_parts[3])
            label = float(bbox_with_label_parts[4])
            tl = np.array([xmin, ymin, 1], np.float32)
            br = np.array([xmax, ymax, 1], np.float32)
            tl = np.dot(amat, tl)
            br = np.dot(amat, br)
            xmin, ymin = tl[0], tl[1]
            xmax, ymax = br[0], br[1]
            xmins.append(xmin / expected_size[0])
            ymins.append(ymin / expected_size[1])
            xmaxs.append(xmax / expected_size[0])
            ymaxs.append(ymax / expected_size[1])

            labels.append(label)

        img = img.astype(np.float32) / 255.
        bbox = np.stack((xmins, ymins, xmaxs, ymaxs, labels), axis=1)
        paddings = [[0, 100 - bbox.shape[0]], [0, 0]]
        bbox = np.pad(bbox, paddings, mode='constant')
        return img, bbox

    def tf_transform_target(img, line):
        img, mask = tf.py_function(func=transform_target, inp=[img, line, dst_size], Tout=[tf.float32, tf.float32])
        img.set_shape((*dst_size[::-1], 1))
        mask.set_shape((100, 5))
        return img, mask

    img_dataset = tf.data.Dataset.from_tensor_slices(imgs)
    bbox_dataset = tf.data.TextLineDataset(os.path.join(data_rootdir, 'bbox.txt'))
    dataset = tf.data.Dataset.zip((img_dataset, bbox_dataset))
    dataset = dataset.map(lambda x, y: tf_transform_target(x, y))
    train_dataset = dataset.take(num_train)
    val_dataset = dataset.skip(num_train)
    return train_dataset, val_dataset