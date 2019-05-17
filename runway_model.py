from __future__ import absolute_import, division, print_function
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from libs.models import *
from libs.utils import DenseCRF

from demo import *
import runway



classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'street sign', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'hat', 26: 'backpack', 27: 'umbrella', 28: 'shoe', 29: 'eye glasses', 30: 'handbag', 31: 'tie', 32: 'suitcase', 33: 'frisbee', 34: 'skis', 35: 'snowboard', 36: 'sports ball', 37: 'kite', 38: 'baseball bat', 39: 'baseball glove', 40: 'skateboard', 41: 'surfboard', 42: 'tennis racket', 43: 'bottle', 44: 'plate', 45: 'wine glass', 46: 'cup', 47: 'fork', 48: 'knife', 49: 'spoon', 50: 'bowl', 51: 'banana', 52: 'apple', 53: 'sandwich', 54: 'orange', 55: 'broccoli', 56: 'carrot', 57: 'hot dog', 58: 'pizza', 59: 'donut', 60: 'cake', 61: 'chair', 62: 'couch', 63: 'potted plant', 64: 'bed', 65: 'mirror', 66: 'dining table', 67: 'window', 68: 'desk', 69: 'toilet', 70: 'door', 71: 'tv', 72: 'laptop', 73: 'mouse', 74: 'remote', 75: 'keyboard', 76: 'cell phone', 77: 'microwave', 78: 'oven', 79: 'toaster', 80: 'sink', 81: 'refrigerator', 82: 'blender', 83: 'book', 84: 'clock', 85: 'vase', 86: 'scissors', 87: 'teddy bear', 88: 'hair drier', 89: 'toothbrush', 90: 'hair brush', 91: 'banner', 92: 'blanket', 93: 'branch', 94: 'bridge', 95: 'building-other', 96: 'bush', 97: 'cabinet', 98: 'cage', 99: 'cardboard', 100: 'carpet', 101: 'ceiling-other', 102: 'ceiling-tile', 103: 'cloth', 104: 'clothes', 105: 'clouds', 106: 'counter', 107: 'cupboard', 108: 'curtain', 109: 'desk-stuff', 110: 'dirt', 111: 'door-stuff', 112: 'fence', 113: 'floor-marble', 114: 'floor-other', 115: 'floor-stone', 116: 'floor-tile', 117: 'floor-wood', 118: 'flower', 119: 'fog', 120: 'food-other', 121: 'fruit', 122: 'furniture-other', 123: 'grass', 124: 'gravel', 125: 'ground-other', 126: 'hill', 127: 'house', 128: 'leaves', 129: 'light', 130: 'mat', 131: 'metal', 132: 'mirror-stuff', 133: 'moss', 134: 'mountain', 135: 'mud', 136: 'napkin', 137: 'net', 138: 'paper', 139: 'pavement', 140: 'pillow', 141: 'plant-other', 142: 'plastic', 143: 'platform', 144: 'playingfield', 145: 'railing', 146: 'railroad', 147: 'river', 148: 'road', 149: 'rock', 150: 'roof', 151: 'rug', 152: 'salad', 153: 'sand', 154: 'sea', 155: 'shelf', 156: 'sky-other', 157: 'skyscraper', 158: 'snow', 159: 'solid-other', 160: 'stairs', 161: 'stone', 162: 'straw', 163: 'structural-other', 164: 'table', 165: 'tent', 166: 'textile-other', 167: 'towel', 168: 'tree', 169: 'vegetable', 170: 'wall-brick', 171: 'wall-concrete', 172: 'wall-other', 173: 'wall-panel', 174: 'wall-stone', 175: 'wall-tile', 176: 'wall-wood', 177: 'water-other', 178: 'waterdrops', 179: 'window-blind', 180: 'window-other', 181: 'wood'}
classes_list = [c for c in classes.values()]


def inference2(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.detach().cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap


def run_model(model, inputs):
    image = np.array(inputs['image'])
    image, raw_image = preprocessing(image, model['device'], model['config'])
    labelmap = inference2(model['model'], image, raw_image, model['postprocessor'])
    return labelmap


@runway.setup(options={'checkpoint': runway.file(extension='.pth')})
def setup(opts):
    config_path = 'configs/cocostuff164k.yaml'
    model_path = opts['checkpoint'] 

    cuda = torch.cuda.is_available()
    crf = False

    with open(config_path, 'r') as f:
        CONFIG = Dict(yaml.load(f))

    device = get_device(cuda)
    torch.set_grad_enabled(False)

    #classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    return Dict({'model': model, 'device': device, 'config': CONFIG, 'postprocessor':postprocessor})




@runway.command('mask_all', inputs={'image': runway.image}, outputs={'image': runway.image})
def mask_all(model, inputs):
    labelmap = run_model(model, inputs)
    image_out = np.dstack([labelmap] * 3).astype(np.uint8)
    return {'image': image_out }


@runway.command('mask_one', inputs={'image': runway.image, 'class': runway.category(choices=classes_list)}, outputs={'image': runway.image})
def mask_one(model, inputs):
    labelmap = run_model(model, inputs)
    labelmap = 255.0 * np.array(labelmap==classes_list.index(inputs['class']))
    image_out = np.dstack([labelmap] * 3).astype(np.uint8)
    return {'image': image_out }


@runway.command('detect', inputs={'image': runway.image}, outputs={'classes': runway.array(runway.text)})
def detect(model, inputs):
    labelmap = run_model(model, inputs)
    labels = [classes_list[l] for l in np.unique(labelmap)]
    return {'classes': labels }


if __name__ == '__main__':
    runway.run()

