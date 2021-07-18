from typing import List, Dict

import cv2
import torch
import torch.backends.cudnn as cudnn

from utils.utils.torch_utils import select_device
from utils.models.models import *
from utils.utils.datasets import *
from utils.utils.general import *


def load_classes(path: str) -> List:
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def detect(cfg, weight, names: str, image: str, classes: int, augment: bool = False, agnostic_nms: bool = False,
           device: str = '', img_size: int = 608, conf_thres: int = 0.1, iou_thres: int = 0.1, save_img: bool = True,
           webcam: bool = False) -> Dict:
    """

    :param cfg: model config file(*.cfg) path
    :param weight: model checkpoint file(*.pt) path
    :param names: class name file(*.names) path
    :param image: detected image path
    :param classes: filter by class: --class 0, or --class 0 2 3
    :param augment: augmented inference
    :param agnostic_nms: class-agnostic NMS
    :param device: cuda device, i.e. 0 or 0,1,2,3 or cpu(default '' -> cpu)
    :param img_size: inference size or image size (pixels)
    :param conf_thres: object confidence threshold, default = 0.1
    :param iou_thres: IOU threshold for NMS, default = 0.1
    :param save_img: whether to save image, default = False
    :param webcam:
    :return:
    """

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
        print(e)
        model.load_state_dict(torch.load(weight, map_location=device))

    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Get names
    names = load_classes(names)

    # load image(BGR)
    img0 = cv2.imread(image)

    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert (BGR to RGB)
    img = img[:, :, ::-1].transpose(2, 0, 1)
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
    results = {'names': names}
    for i, det in enumerate(pred):
        results['prediction'] = []
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh2(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                results['prediction'].append((*xywh, conf.data.item(), cls.data.item() + 1, 0, 0))

            return results
        else:
            results['prediction'].append((0, 0, 0, 0, -1, -1, 0, 0))
            return results


if __name__ == "__main__":
    detect('./cfg/295_sparsity_5_prune_0.5_0.5.cfg',
           './cfg/t-295-s-5-p-0.5-0.5-ft-500-e-784-f-784.pt',
           './cfg/visDrone.names',
           './cfg/0000013_00465_d_0000067.jpg',
           0)
