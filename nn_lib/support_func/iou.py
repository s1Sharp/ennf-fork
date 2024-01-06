#from nn_lib.tensor import Tensor

import numpy as np

def iou_score(outputs, labels):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.data.squeeze().astype(np.uint8)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.data.squeeze().astype(np.uint8)
    SMOOTH = 1e-8
    intersection = (outputs & labels).astype(np.float).sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).astype(np.float).sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10  # This is equal to comparing with thresolds

    return thresholded  #