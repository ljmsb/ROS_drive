import os
import cv2
import argparse
from keras.models import load_model
from keras import backend as K
import numpy as np
import time
from box import to_minmax
from box import BoundBox, nms_boxes, boxes_to_array,draw_scaled_boxes
import os
os.environ["TF_KERAS"] = '1'


labels = ["yellow","green","red","stop","hand","turn right","thing"]
input_size= [224,224]
anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
nms_threshold = 0.2
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

def normalize(image):
    image = image / 255.
    image = image - 0.5
    image = image * 2.

    return image

def get_lable_score(probs,labels):
    label_score_list = []
    for classes in probs:
        label_score_list.append({'label':labels[np.argmax(classes)] ,'score': classes.max()})
    return label_score_list

def prepare_image(orig_image):

    input_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (input_size[1],input_size[0]))
    input_image = normalize(input_image)
    input_image = np.expand_dims(input_image, 0)
    return orig_image, input_image

def run(netout, obj_threshold=0.3):

    grid_h, grid_w, nb_box = netout.shape[:3]
    boxes = []

    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):

                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:

                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + _sigmoid(x)) / grid_w
                    y = (row + _sigmoid(y)) / grid_h
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h
                    confidence = netout[row, col, b, 4]
                    box = BoundBox(x, y, w, h, confidence, classes)
                    boxes.append(box)

    boxes = nms_boxes(boxes, len(classes), nms_threshold, obj_threshold)
    boxes, probs = boxes_to_array(boxes)
    return boxes, probs

def predict(model, image, height, width, threshold=0.3):


    def _to_original_scale(boxes):
        # height, width = image.shape[:2]
        minmax_boxes = to_minmax(boxes)
        minmax_boxes[:, 0] *= width
        minmax_boxes[:, 2] *= width
        minmax_boxes[:, 1] *= height
        minmax_boxes[:, 3] *= height
       
        return minmax_boxes.astype(np.int)

    start_time = time.time()
    netout = model.predict(image)[0]
    elapsed_ms = (time.time() - start_time) * 1000
    boxes, probs = run(netout, threshold)
    if len(boxes) > 0:
        boxes = _to_original_scale(boxes)
        #print(boxes, probs)
        return elapsed_ms, boxes, probs
    else:
        return elapsed_ms, [], []


def test_camera():

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    model_path = 'YOLO_best_mAP.h5'
    #'/home/deepcar/deepcar_py2/selfdriving_object_detection/traffic_signs_YOLO_best_mAP.h5'
    model = load_model(model_path)

    while (1):

        ret, frame = cap.read()
        orig_image, input_image = prepare_image(frame)
        height, width = orig_image.shape[:2]
        prediction_time, boxes, probs = predict(model, input_image, height, width)
        print(prediction_time)
        orig_image = draw_scaled_boxes(orig_image, boxes, probs, labels)
        cv2.imshow("capture", orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_camera()
