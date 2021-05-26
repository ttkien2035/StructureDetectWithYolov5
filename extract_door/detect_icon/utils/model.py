import cv2
import numpy as np
import time
import torch

from detect_icon.models.experimental import attempt_load
from detect_icon.utils.datasets import letterbox
from detect_icon.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from detect_icon.utils.torch_utils import select_device, time_synchronized

def load_model(weights_path, device):
    # Initialize
    set_logging()
    device  = select_device(device)

    # Load model
    model = attempt_load(weights_path, map_location=device)  # load FP32 model

    print('device= ', device)
    print('Load model successfully.')
    return model, device

def detect(img_path, model, device, imgsz = 640, conf = 0.7, iou_thresh = 0.45):
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    tmp = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(tmp.half() if half else tmp) if device.type != 'cpu' else None  # run once

    # Padded resize
    if isinstance(img_path, str):
        img0 = cv2.imread(img_path)
    else:
        img0 = img_path
    img = letterbox(img0, new_shape=imgsz)[0]
#     print(img.shape)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf, iou_thresh, classes=None, agnostic=True)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = img0.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#                 print(det)

    # print(f'Done. ({time.time() - t0:.3f}s)')
    res_img = img0.copy()
    for xmin, ymin, xmax, ymax, prob, name in det:
        res_img = cv2.rectangle(res_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,0,0), 1)
    return det, img0, res_img