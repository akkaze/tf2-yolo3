from absl import logging
import numpy as np
import tensorflow as tf
import cv2


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names=None):
    boxes, objectness, classes = outputs
    boxes, objectness, classes = boxes[0], objectness[0], classes[0]
    wh = np.flip(img.shape[0:2])
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    min_wh = np.amin(wh)
    if min_wh <= 100:
        font_size = 1
    else:
        font_size = 1
    for i in range(classes.shape[0]):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 1)
        img = cv2.putText(img, '{}'.format(int(classes[i])), x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size,
                          (0, 0, 255), 1)
    return img


def draw_labels(x, y, class_names=None):
    img = x.numpy()
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 1)
        if class_names:
            img = cv2.putText(img, class_names[classes[i]], (x1y1[0] - 2, x1y1[1] - 2), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                              0.4, (0, 0, 255), 1)
        else:
            img = cv2.putText(img, str(classes[i]), x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def cv2_letterbox_resize(img, expected_size):
    ih, iw = img.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    smat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    tmat = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]], np.float32)
    amat = np.dot(tmat, smat)
    amat_ = amat[:2, :]
    dst = cv2.warpAffine(img, amat_, expected_size)
    if dst.ndim == 2:
        dst = np.expand_dims(dst, axis=-1)
    return dst, amat


def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    req = urllib.request.urlopen(url)
    file_size = int(req.info().get('Content-Length', -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(total=file_size, initial=first_byte, unit='B', unit_scale=True, desc=url.split('/')[-1])
    with (open(dst, 'ab')) as f:
        while True:
            chunk = req.read(1024)
            if not chunk:
                break
            file_size += len(chunk)
            f.write(chunk)
            pbar.update(len(chunk))
    pbar.close()
    return file_size