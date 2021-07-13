import os
import argparse
from pathlib import Path
from typing import List, Dict

import cv2
from numpy import random
import torch
import torch.backends.cudnn as cudnn

from utils.models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from models.experimental import *
from utils.datasets import *
from utils.general import *


def load_classes(path: str) -> List:
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


#     parser.add_argument('--weights', nargs='+', type=str, default='./weights/last_prune_0593.pt',
#                         help='model.pt path(s)')
#     parser.add_argument('--cfg', type=str, default='./cfg/295_sparsity_5_prune_0.5_0.5.cfg', help='*.cfg path')
#     parser.add_argument('--output', type=str, default='./output', help='output folder')  # output folder
#     parser.add_argument('--names', type=str, default='data/visDrone.names', help='*.cfg path')
#     parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', default=True, type=bool, help='save results to *.txt')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')

def detect(cfg, weight, names: str, img_path, classes: int, augment: bool = False, agnostic_nms: bool = False, device: str = '',
           img_size: int = 608, conf_thres: int = 0.1, iou_thres: int = 0.1,
           save_img: bool = True, webcam: bool = False) -> Dict:
           
    # select device (cpu or cuda)
    device = select_device(device)

    # half precision only supported on CUDA
    half = device.type != 'cpu'

    # Load model
    model = Darknet(cfg, img_size).to(device)

    # load state dict.
    try:
        state_dict = torch.load(weight, map_location=device)['model']
        model.load_state_dict(state_dict)
    except TypeError as e:
        model.load_state_dict(torch.load(weight, map_location=device))

    model.to(device).eval()
    if half:
        model.half()  # to FP16

    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Get names
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # load image
    img0 = cv2.imread(img_path)  # BGR
    assert img0 is not None, 'Image Not Found ' + img_path

    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x608x608
    img = np.ascontiguousarray(img)

    # detect
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes,
                               agnostic=agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):
        if webcam:
            p, s, im0 = img_path[i], '%g: ' % i, img0[i].copy()  # batch_size >= 1
        else:
            p, s, im0 = img_path, '', img0

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh2(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    with open('./111' + '.txt', 'a') as f:
                        f.write(('%g ' * 8 + '\n') % (*xywh, conf, (cls + 1), 0, 0))  # label format

                    if save_img:  # Add bbox to image
                        plot_one_box(xyxy, im0, label=None, color=colors[int(cls)], line_thickness=2)
            else:
                with open('./111' + '.txt', 'a') as f:
                    f.write(('%g ' * 8 + '\n') % (0, 0, 0, 0, -1, -1, 0, 0))  # label format

            # Save results (image with detections)
            if save_img:
                cv2.imwrite('./111.png', im0)

            # if save_txt or save_img:
            #     print('Results saved to %s' % Path(out))
            #     if platform == 'darwin' and not opt.update:  # MacOS
            #         os.system('open ' + save_path)

if __name__ == "__main__":
    detect('./cfg/295_sparsity_5_prune_0.5_0.5.cfg',
           './cfg/t-295-s-5-p-0.5-0.5-ft-500-e-784-f-784.pt',
           './cfg/visDrone.names',
           './cfg/0000013_00465_d_0000067.jpg',
           0)