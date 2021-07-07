import argparse
from models.models import Darknet
from typing import List
import torch
import torchsummary

def parameters(weights: str, cfg: str, img_size: int, channel: int = 3) -> None:
    if not weights.endswith('.pt') or weights == '' or weights == None:
        raise ValueError("weights is not support")
    elif not cfg.endswith('.cfg') or cfg == '' or cfg == None:
        raise ValueError("cfg is not support")

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Darknet(cfg=cfg, img_size=img_size)#.to(device)

    torchsummary.summary(model, (channel, img_size[0], img_size[1]), batch_size=1, device='cpu')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/last_299_spk_spp.pt', help='initial weights path or pre-training model state dict (.pt/.pth)')
    parser.add_argument('--cfg', type=str, default='./cfg/295_sparsity_5_prune_0.5_0.5.cfg', help='model .cfg file path') #295_sparsity_5_prune_0.5_0.5    yolov4-spk-spp.cfg
    parser.add_argument('--img-size', type=int, default=[608, 608], help='the input of model image size')
    opt = parser.parse_args()

    parameters(opt.weights, opt.cfg, opt.img_size, channel=3)