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

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    return Dict({'model': model, 'device': device, 'config': CONFIG, 'postprocessor':postprocessor})


@runway.command('convert', inputs={'image': runway.image}, outputs={'image': runway.image})
def convert(model, inputs):
    image_in = np.array(inputs['image'])
    image, raw_image = preprocessing(image_in, model['device'], model['config'])
    labelmap = inference2(model['model'], image, raw_image, model['postprocessor'])
    labels = np.unique(labelmap)

    # Show result for each class
    rows = np.floor(np.sqrt(len(labels) + 1))
    cols = np.ceil((len(labels) + 1) / rows)
    
    image_out = np.dstack([labelmap]*3).astype(np.uint8)
    return {'image': image_out }


#curl -X POST --data @exam.json -H 'Content-Type: application/json' http://localhost:8000/generate


if __name__ == '__main__':
    runway.run()

