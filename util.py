import os
import cv2
import numpy as np
import onnxruntime as ort


def distance2box(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    outputs = []
    for i in range(0, distance.shape[1], 2):
        p_x = points[:, i % 2] + distance[:, i]
        p_y = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            p_x = p_x.clamp(min=0, max=max_shape[1])
            p_y = p_y.clamp(min=0, max=max_shape[0])
        outputs.append(p_x)
        outputs.append(p_y)
    return np.stack(outputs, axis=-1)


class FaceDetector:
    def __init__(self, onnx_path=None, session=None):
        from onnxruntime import InferenceSession
        self.session = session

        self.batched = False
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])
        self.nms_thresh = 0.4
        self.center_cache = {}
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for output in outputs:
            output_names.append(output.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, x, score_thresh):
        scores_list = []
        bboxes_list = []
        points_list = []
        input_size = tuple(x.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(x,
                                     1.0 / 128,
                                     input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = outputs[idx][0]
                boxes = outputs[idx + fmc][0]
                boxes = boxes * stride
            else:
                scores = outputs[idx]
                boxes = outputs[idx + fmc]
                boxes = boxes * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1)
                anchor_centers = anchor_centers.astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1)
                    anchor_centers = anchor_centers.reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_indices = np.where(scores >= score_thresh)[0]
            bboxes = distance2box(anchor_centers, boxes)
            pos_scores = scores[pos_indices]
            pos_bboxes = bboxes[pos_indices]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
        return scores_list, bboxes_list

    def detect(self, image, input_size=None, score_threshold=0.5, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        image_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if image_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / image_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * image_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list = self.forward(det_img, score_threshold)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            index = np.argsort(values)[::-1]  # some extra weight on the centering
            index = index[0:max_num]
            det = det[index, :]
        return det

    def nms(self, outputs):
        thresh = self.nms_thresh
        x1 = outputs[:, 0]
        y1 = outputs[:, 1]
        x2 = outputs[:, 2]
        y2 = outputs[:, 3]
        scores = outputs[:, 4]

        order = scores.argsort()[::-1]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= thresh)[0]
            order = order[indices + 1]

        return keep


class HSEmotionRecognizer:
    def __init__(self, path):
        self.img_size = 260
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.idx_to_class = {0: 'ANGER',
                             1: 'DISGUST',
                             2: 'FEAR',
                             3: 'HAPPINESS',
                             4: 'NEUTRAL',
                             5: 'SADNESS',
                             6: 'SURPRISE'}

        self.ort_session = ort.InferenceSession(path, providers=['CUDAExecutionProvider'])

    def preprocess(self, img):
        x = cv2.resize(img, (self.img_size, self.img_size)) / 255
        for i in range(3):
            x[..., i] = (x[..., i] - self.mean[i]) / self.std[i]
        return x.transpose(2, 0, 1).astype("float32")[np.newaxis, ...]

    def predict_emotions(self, face_img, logits=True):
        scores = self.ort_session.run(None, {"input": self.preprocess(face_img)})[0][0]
        x = scores
        pred = np.argmax(x)
        if not logits:
            e_x = np.exp(x - np.max(x)[np.newaxis])
            e_x = e_x / e_x.sum()[None]
            scores = e_x
        return self.idx_to_class[pred], scores

    def predict_multi_emotions(self, face_img_list, logits=True):
        images = np.concatenate([self.preprocess(face_img) for face_img in face_img_list], axis=0)
        scores = self.ort_session.run(None, {"input": images})[0]
        pred = np.argmax(scores, axis=1)
        x = scores

        if not logits:
            e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:, None]
            scores = e_x

        return [self.idx_to_class[pred] for pred in pred], scores


def draw_rounded_bar(image, top_left, bottom_right, color, radius):
    top_left = (int(top_left[0]), int(top_left[1]))
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
    image_height, image_width = image.shape[:2]

    bottom_right = (min(bottom_right[0], image_width), min(bottom_right[1], image_height))

    cv2.rectangle(image, (top_left[0] + radius, top_left[1]), (bottom_right[0] - radius, bottom_right[1]), color,
                  thickness=2)
    cv2.rectangle(image, (top_left[0], top_left[1] + radius), (bottom_right[0], bottom_right[1] - radius), color,
                  thickness=2)

    cv2.circle(image, (top_left[0] + radius, top_left[1] + radius), radius, color, -1)
    cv2.circle(image, (bottom_right[0] - radius, top_left[1] + radius), radius, color, -1)
    cv2.circle(image, (top_left[0] + radius, bottom_right[1] - radius), radius, color, -1)
    cv2.circle(image, (bottom_right[0] - radius, bottom_right[1] - radius), radius, color, -1)


def draw_emotion_bars(model, frame, scores, top_left, bar_height=25, width=150, spacing=10):
    emotions = list(model.idx_to_class.values())
    start_x, start_y = top_left

    overlay = frame.copy()
    overlay_height = len(emotions) * (bar_height + spacing) - spacing
    cv2.rectangle(overlay, (start_x, start_y), (start_x + width, start_y + overlay_height), (0, 0, 0, 128), -1)
    cv2.addWeighted(overlay, 0.4, frame, 1 - 0.4, 0, frame)

    for i, (emotion, score) in enumerate(zip(emotions, scores)):
        bar_length = int(score * width)
        bar_top_left = (start_x, start_y + i * (bar_height + spacing))
        bar_bottom_right = (start_x + bar_length, start_y + (i * (bar_height + spacing)) + bar_height)
        draw_rounded_bar(frame, bar_top_left, bar_bottom_right, (0, 255, 0), radius=int(bar_height / 3))

        text_position = (start_x + width + 5, bar_top_left[1] + bar_height // 2 + spacing // 4)
        cv2.putText(frame, emotion, text_position, cv2.FONT_HERSHEY_COMPLEX, 0.4, (66, 15, 7), 1)
