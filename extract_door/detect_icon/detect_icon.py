from pickle import load
import cv2
import torch
import os, sys

from detect_icon.models.experimental import attempt_load
from detect_icon.utils.model import select_device
from detect_icon.utils.preprocess import preprocess
from detect_icon.utils.box import remove_icon, remove_icon_2, get_icon_boxes, \
                            get_icon_boxes_with_class, remove_duplicate_boxes
from utils import SingletonMeta
from config import DETECT_ICON__WEIGHT__PATH, INPUT_SIZE, CLASSES

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class IconDetector(metaclass=SingletonMeta):
    def __init__(self) -> None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        self.model = attempt_load(DETECT_ICON__WEIGHT__PATH,
                                  map_location=device)
        del sys.path[0]
        self.device = select_device(device)

    def predict(self, org_img):
        img_gray, org_img = preprocess(org_img, 'gray')
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        raw_icon_boxes =  get_icon_boxes((self.model, self.device), img_gray,
                                         INPUT_SIZE, CLASSES)
        final_icon_boxes = remove_duplicate_boxes(raw_icon_boxes)

        return final_icon_boxes

    def predict_with_class(self, org_img):
        img_gray, org_img = preprocess(org_img, 'gray')
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        raw_icon_boxes, raw_icon_door = get_icon_boxes_with_class((self.model, self.device), img_gray,
                                         INPUT_SIZE, CLASSES)
        final_icon_boxes = remove_duplicate_boxes(raw_icon_boxes)
        final_icon_door = remove_duplicate_boxes(raw_icon_door)

        return final_icon_boxes, final_icon_door

    @classmethod
    def remove_icon(self, org_img, icon_boxes):
        output_img, icon_mask = remove_icon_2(org_img, icon_boxes)

        return output_img, icon_mask

if __name__ == '__main__':
    print('abc')