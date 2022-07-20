from typing import List, Tuple
import _init_paths
import os
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import (
    # save_net, load_net, 
    # vis_detections, 
    # vis_detections_PIL, 
    vis_detections_filtered_objects_PIL, 
    # vis_detections_filtered_objects
)  # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import tqdm


import contextlib
@contextlib.contextmanager
def _timer(*a):
    t0=time.time()
    yield 
    print(*a, f"took {time.time() - t0:.4g}s", flush=True)

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY



# WEIGHTS = {}



cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

class FasterRCNN(nn.Module):
    object_classes = ['targetobject', 'hand']
    default_threshold = 0.5
    def __init__(
        self, 
        checkpoint_path: str, 
        net='res101', 
        class_agnostic=False,
        thresh_obj=0.5, 
        thresh_hand=0.5,
    ):
        super().__init__()
        self.class_agnostic = class_agnostic
        self.thresholds = [thresh_obj, thresh_hand]

        # initilize the network here.
        self.pascal_classes = pascal_classes = np.array(['__background__'] + list(self.object_classes))
        if net == 'vgg16':
            fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=class_agnostic)
        elif net == 'res101':
            fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=class_agnostic)
        elif net == 'res50':
            fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=class_agnostic)
        elif net == 'res152':
            fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=class_agnostic)
        else:
            raise ValueError(f"network {net} is not defined")
        fasterRCNN.create_architecture()
        self.model = fasterRCNN

        if checkpoint_path:
            print(f"load checkpoint {checkpoint_path}")
            if cuda:
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=(lambda storage, loc: storage))
            fasterRCNN.load_state_dict(checkpoint['model'])
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']
            print('load model successfully!')
        
        fasterRCNN.to(device)
        fasterRCNN.eval()

        cfg.CUDA = cuda
        # args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

    @classmethod
    def from_model_dir(cls, checkpoint, load_dir='models', net='res101', dataset='pascal_voc', checksession=1, checkepoch=8, **kw):
        # load model
        model_dir = f"{load_dir}/{net}_handobj_100K/{dataset}"
        if not os.path.exists(model_dir):
            raise Exception(f'There is no input directory for loading network from {model_dir}')
        checkpoint_path = os.path.join(model_dir, f'faster_rcnn_{checksession}_{checkepoch}_{checkpoint}.pth')
        return cls(checkpoint_path, **kw)


    def forward(self, im: np.ndarray):x
        im_data, im_info, im_scales = self.preprocess(im)
        output = self.predict(im_data, im_info, im_scales)
        return output


    def preprocess(self, im):
        im_blob, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(
            np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32))
        return im_data_pt, im_info_pt, im_scales

    def predict(self, im_data, im_info, im_scales):
        gt_boxes = torch.zeros((1, 1, 5))
        num_boxes = torch.zeros((1,))
        box_info = torch.zeros((1, 1, 5))
        (
            # tensors: (1, 300, 5), (1, 300, 3), (1, 300, 12)
            rois, cls_prob, bbox_pred,
            # ints 
            rpn_loss_cls, rpn_loss_box,
            RCNN_loss_cls, RCNN_loss_bbox,
            # None
            rois_label, 
            # extact predicted params
            # - hand contact state info
            # - offset vector (factored into a unit vector and a magnitude)
            # - hand side info (left/right)
            # tensors: [( (1, 300, 5), int ), ( (1, 300, 3), int ), ( (1, 300, 1), int )]
            ((contact_vector, _), (offset_vector, _), (lr_vector, _))
        ) = self.model(im_data, im_info, gt_boxes, num_boxes, box_info)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # get hand contact 
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices[0][..., None].float()
        # get hand side 
        lr = (torch.sigmoid(lr_vector.detach()) > 0.5)[0].float()

        if cfg.TEST.BBOX_REG:  # Apply bounding-box regression deltas
            pred_boxes = self._bbox_regression_deltas(bbox_pred, boxes, im_info)
        else:  # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        pred_boxes /= im_scales[0]

        scores = scores[0]
        pred_boxes = pred_boxes[0]
        offset_vector = offset_vector.detach()[0]
        return [
            self._nms(
                scores[:,j], 
                pred_boxes if self.class_agnostic else pred_boxes[:, j * 4:(j + 1) * 4] , 
                contact_indices, 
                offset_vector, 
                lr,
                (self.thresholds[j] if j < len(self.thresholds) else None) or self.default_threshold)
            for j in range(len(self.pascal_classes))
        ]

    def _nms(self, scores, pred_boxes, contact_indices, offset_vector, lr, thresh):
        inds = torch.nonzero(scores > thresh).view(-1)
        cls_scores = scores[inds]
        cls_boxes = pred_boxes[inds]
        _, order = torch.sort(cls_scores, 0, True)
        keep = nms(cls_boxes[order], cls_scores[order], cfg.TEST.NMS)
        cls_dets = torch.cat((
            cls_boxes, 
            cls_scores[:, None], 
            contact_indices[inds], 
            offset_vector[inds], 
            lr[inds]), 1)
        cls_dets = cls_dets[order][keep.view(-1).long()]
        return cls_dets

    def _bbox_regression_deltas(self, bbox_pred, boxes, im_info):
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = (
                box_deltas.view(-1, 4) * 
                torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) + 
                torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
            ).view(1, -1, 4 * (1 if self.class_agnostic else len(self.pascal_classes)))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        return pred_boxes


def _get_image_blob(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a BGR image into an image pyramid blob with associated image scales.

    Arguments:
        im (np.ndarray): a color image in BGR order

    Returns:
        blob (np.ndarray): a data blob holding an image pyramid
        im_scale_factors (np.ndarray[(n_scales,), int]): 
            image scales (relative to im) in the image pyramid
    """
    im_orig = im - cfg.PIXEL_MEANS
    # Create a blob to hold the input images
    scales = _get_image_blob_scales(im.shape, cfg.TEST.SCALES, cfg.TEST.MAX_SIZE)
    blob = im_list_to_blob([
        cv2.resize(im_orig, shape, interpolation=cv2.INTER_LINEAR) 
        for shape in scales
    ])
    return blob, np.array([h/im.shape[0] for h, _ in scales])


def _get_image_blob_scales(shape, scales: list, max_size: float):
    h, w = shape[:2]
    hw, wh = (1, w/h) if h > w else (h/w, 1)
    return [
        (int(h), int(w)) for h, w in (
            (max_size, w / h * max_size) if h > max_size else 
            (h / w * max_size, max_size) if w > max_size else 
            (h, w) for h, w in ((hw * s, wh * s) for s in scales)
        )
    ]


class ImageLoader:
    fps = 10
    def __init__(self, src):
        self.src = src
        if isinstance(self.src, str) and os.path.isdir(self.src):
            self.it = self._dir(self.src)
        else:
            self.it = self._video(self.src)

    def __iter__(self):
        return self.it

    def _video(self, src):
        cap = cv2.VideoCapture(src)
        self.total = total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        def _gen():
            try:
                i = 0
                self.pbar = pbar = tqdm.tqdm(total=int(total) if total else None)
                while True:
                    if not cap.isOpened():
                        raise RuntimeError("Webcam could not open. Please check connection.")
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError("No frame returned. Please check connection.")
                    yield f'{i:04d}-{src}', np.array(frame)
                    i += 1
                    pbar.update()
            finally:
                cap.release()
        return _gen()

    def _dir(self, src):
        fs = os.listdir(src)
        self.pbar = tqdm.tqdm(fs)
        self.total = len(fs)
        def _gen():
            for f in self.pbar:
                yield f, cv2.imread(os.path.join(src, f))
        return _gen()


class ImageWriter:
    def __init__(self, src, fps, cc='avc1') -> None:
        self.src = src
        self.cc = cc
        self.fps = fps

    def __enter__(self): pass

    def __exit__(self, *a):
        if self._w:
            self._w.release()

    _w = None
    def write_video(self, im):
        if not self._w:
            self._w = cv2.VideoWriter(
                self.src, cv2.VideoWriter_fourcc(*self.cc), 
                self.fps, im.shape[:2][::-1], True)
        self._w.write(im)

    def write_frame(self, im, name):
        im.save(os.path.join(self.src, f"{os.path.splitext(name)[0]}_det.png"))

# def image_loader(src):
#     if isinstance(src, int):
#         cap = cv2.VideoCapture(src)
#         try:
#             i = 0
#             pbar = tqdm.tqdm()
#             while True:
#                 if not cap.isOpened():
#                     raise RuntimeError("Webcam could not open. Please check connection.")
#                 ret, frame = cap.read()
#                 if not ret:
#                     raise RuntimeError("No frame returned. Please check connection.")
#                 yield f'{i:04d}-{src}', np.array(frame)
#                 i += 1
#                 pbar.update()
#         finally:
#             cap.release()
#             cv2.destroyAllWindows()
#     else:
#         for f in tqdm.tqdm(os.listdir(src)):
#             im_file = os.path.join(src, f)
#             yield f, cv2.imread(im_file)




def main(src, save_dir=None, show=None, cfg_file='cfgs/res101.yml', set_cfgs=None, checkpoint='132028'):
    show = not save_dir if show is None else show
    if cfg_file is not None:
        cfg_from_file(cfg_file)
    set_cfgs = list(set_cfgs or ()) + ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)
    cfg.USE_GPU_NMS = cuda
    np.random.seed(cfg.RNG_SEED)

    if save_dir: 
        os.makedirs(save_dir, exist_ok=True)

    model = FasterRCNN.from_model_dir(checkpoint)
    # thresh_obj, thresh_hand = model.thresholds

    try:
        loader = ImageLoader(src)
        with ImageWriter(save_dir, loader.fps) as writer:
            for f, im in loader:
                _, obj_dets, hand_dets = model(im)

                im2show = vis_detections_filtered_objects_PIL(
                    np.copy(im), obj_dets, hand_dets, 0, 0)#thresh_hand, thresh_obj
                if show:
                    cv2.imshow("frame", cv2.cvtColor(np.array(im2show), cv2.COLOR_BGR2RGB))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if save_dir:
                    im2show.save(os.path.join(save_dir, f"{os.path.splitext(f)[0]}_det.png"))
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import fire
    fire.Fire(main)