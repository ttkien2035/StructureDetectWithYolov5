import cv2
import numpy as np

from detect_icon.utils.model import detect
from detect_icon.utils.preprocess import get_table_mask, separate_image

def map_box_to_orginal_image(box_in_small, small_in_big):
    """
        -> box_in_small: bounding box in the `cropped image`
        -> small_in_big: bounding box/coordinates of the small/cropped image in large image
        -> boxes in format [xmin,ymin,xmax,ymax]
        -> example:
            >>> box_in_small = [200,200,300,300]
            >>> small_in_big = [250,250,750,750]
            >>> box_in_big = map_box_in_small_to_big(box_in_small, small_in_big)
            >>> box_in_big
            [450, 450, 550, 550]
    """
    return [
        box_in_small[0] + small_in_big[0],
        box_in_small[1] + small_in_big[1],
        box_in_small[2] + small_in_big[0],
        box_in_small[3] + small_in_big[1],
    ]

def get_icon_boxes(models, source, input_size, classes):
    """ Detect and extract icons (raw) from orginal image
        Input:
            -> models: (model, device) used to detect
            -> source: np.array
            -> postition_in_org_img: [xmin, ymin, xmax, ymax]
            -> classes: list
        Return: list bounding boxes of icon 
    """
    #     colors = []
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    overlap = 160
    sub_images, sub_boxes = separate_image(source, input_size, overlap)
    raw_icon_boxes = []
    (model, device) = models
    for i in range(sub_images.shape[0]):
        for j in range(sub_images.shape[1]):
            icon_list = []
            det, _, _ = detect(sub_images[i][j], model, device, 640, .7, .45)
            for xmin, ymin, xmax, ymax, prob, name in det:
                if classes[int(name)] != 'door':
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    icon_list.append(map_box_to_orginal_image([xmin, ymin, xmax, ymax], sub_boxes[i][j]))
                    
            raw_icon_boxes.extend(icon_list)
#             c = get_random_color()
#             colors.extend([c for _ in range(len(icon_list))])

    return raw_icon_boxes


def get_icon_boxes_with_class(models, source, input_size, classes):
    """ Detect and extract icons (raw) from orginal image
        Input:
            -> models: (model, device) used to detect
            -> source: np.array
            -> postition_in_org_img: [xmin, ymin, xmax, ymax]
            -> classes: list
        Return: list bounding boxes of icon
    """
    #     colors = []
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    overlap = 160
    sub_images, sub_boxes = separate_image(source, input_size, overlap)
    raw_icon_boxes = []
    raw_icon_door = []
    (model, device) = models
    for i in range(sub_images.shape[0]):
        for j in range(sub_images.shape[1]):
            icon_list = []
            door_list = []
            det, _, _ = detect(sub_images[i][j], model, device, 640, .7, .45)
            for xmin, ymin, xmax, ymax, prob, name in det:
                if classes[int(name)] != 'door':
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    icon_list.append(map_box_to_orginal_image([xmin, ymin, xmax, ymax], sub_boxes[i][j]))
                else:
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    door_list.append(map_box_to_orginal_image([xmin, ymin, xmax, ymax], sub_boxes[i][j]))
            raw_icon_boxes.extend(icon_list)
            raw_icon_door.extend(door_list)
#             c = get_random_color()
#             colors.extend([c for _ in range(len(icon_list))])

    return raw_icon_boxes, raw_icon_door


def calculate_2way_iou( box1 , box2 ):
    def calc_box_area(box):
        area = (box[2] - box[0]) * (box[3] - box[1])
        return area

    xA = np.maximum( box1[0], box2[0] )
    yA = np.maximum( box1[1], box2[1] )
    xB = np.minimum( box1[2], box2[2] )
    yB = np.minimum( box1[3], box2[3] )
    inter_area = np.maximum(0.0, xB - xA ) * np.maximum(0.0, yB - yA )
    boxA_area  = calc_box_area(box1)
    boxB_area  = calc_box_area(box2)
    uni_area   = ( boxA_area + boxB_area - inter_area )
    iou = inter_area / uni_area
    box1_ratio = inter_area / boxA_area
    box2_ratio = inter_area / boxB_area
    return inter_area, iou, box1_ratio, box2_ratio


def remove_duplicate_boxes(boxes, threshold = 0.8):
    """ Remove overlap boxes after merging small images
        Input:
            -> boxes: list
        
        Return: list bounding boxes of icon 
    """
    ans = []
    keep = [True for _ in range(len(boxes))]
    for i1 in range(len(boxes)):
        for i2 in range(len(boxes)):
            if i1 != i2 and keep[i2] == True:
                _, _, r1, r2 = calculate_2way_iou(boxes[i1], boxes[i2])
                if r1 > threshold:
                    keep[i1] = False
                    break

    for i in range(len(keep)):
        if keep[i]: ans.append(boxes[i])
    
    return ans

def remove_icon(source, icon_boxes):
    """ Remove icon from its bounding boxes by replace its pixels by the mean of background
        Input:
            -> source: np.array
            -> icon_boxes: list
        Return: 
            -> Image after removing icon: np.array
            -> Image with masks of icon: np.array (binary image)
    """
    icon_mask = np.zeros((source.shape[0], source.shape[1]), dtype = np.uint8)
    kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    res_img = source.copy()
    for xmin, ymin, xmax, ymax in icon_boxes:
            xmin, ymin, xmax, ymax = int(xmin)+1, int(ymin)+1, int(xmax)+1, int(ymax)+1
            icon_box = source[ymin:ymax, xmin:xmax, :].astype(np.uint8)
            tmp = icon_box.copy()
            icon_box = cv2.cvtColor(icon_box, cv2.COLOR_BGR2GRAY)
            icon_box = cv2.threshold(icon_box, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            icon_box = ~cv2.dilate(~icon_box, kernel, iterations=1)

            # Replace value of original pixels
            bg_mean_value = tmp[icon_box == 255].mean(axis = 0)
            tmp[icon_box != 255] = bg_mean_value
            res_img[ymin:ymax, xmin:xmax] = tmp
            icon_mask[ymin:ymax, xmin:xmax] = ~icon_box.copy()
#     final_img = cv2.inpaint(final_img, mask, 5 , cv2.INPAINT_NS)

    # Return line pixels
    line_mask = get_table_mask(source)
    x, y = np.argwhere(line_mask == 255).T
    res_img[x, y] = source[x, y]
    return res_img, icon_mask


def remove_icon_2(source, icon_boxes, dilation_kernel_size = 5):
    """ Remove icon from its bounding boxes by replace its pixels by the mean of background
        Input:
            -> source: np.array
            -> icon_boxes: list
            -> dilation_kernel_size : int (must be odd)
        Return:
            -> Image after removing icon: np.array
            -> Image with masks of icon: np.array (binary image)
    """
    icon_mask = np.zeros((source.shape[0], source.shape[1]), dtype = np.uint8)
    kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size))
    res_img = source.copy()
    for xmin, ymin, xmax, ymax in icon_boxes:
            xmin, ymin, xmax, ymax = int(xmin) - 3, int(ymin) - 3, int(xmax) + 3, int(ymax) + 3
            if xmin < 0:
              xmin = 0
            if ymin < 0:
              ymin = 0
            icon_box = source[ymin:ymax, xmin:xmax, :].astype(np.uint8)
            tmp = icon_box.copy()
            icon_box = cv2.cvtColor(icon_box, cv2.COLOR_BGR2GRAY)
            icon_box = cv2.threshold(icon_box, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            dilated_icon_box = ~cv2.dilate(~icon_box, kernel, iterations=1)

            smaller_dilation_kernel_size = dilation_kernel_size
            while not np.isin(255, dilated_icon_box): # Check in case dilation produce only black pixel image
              # print("kernel too big. No 255 value")
              if smaller_dilation_kernel_size > 1: # Check if dilation kernel size > 1, dilate icon_box with smaller kernel
                smaller_dilation_kernel_size = smaller_dilation_kernel_size - 2
                smaller_kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (smaller_dilation_kernel_size, smaller_dilation_kernel_size))
                dilated_icon_box = ~cv2.dilate(~icon_box, smaller_kernel, iterations=1)
              else:
                break

            icon_box = dilated_icon_box.copy()
            # Replace value of original pixels
            bg_mean_value = tmp[icon_box == 255].mean(axis = 0)
            if np.isnan(bg_mean_value).any(): # mask icon box doesn't have value 255
              bg_mean_value = tmp[icon_box != 255].mean(axis = 0)
            tmp[icon_box != 255] = bg_mean_value
            res_img[ymin:ymax, xmin:xmax] = tmp
            icon_mask[ymin:ymax, xmin:xmax] = ~icon_box.copy()
#     final_img = cv2.inpaint(final_img, mask, 5 , cv2.INPAINT_NS)

    # Return line pixels
    line_mask = get_table_mask(source)
    x, y = np.argwhere(line_mask == 255).T
    res_img[x, y] = source[x, y]
    return res_img, icon_mask


def get_random_color(seed=None, n_channels=3):
    """ Get 1 random color in `n_channels` channels
    example:
        >>> get_random_color()
        [183, 228, 163]
        >>> get_random_color()
        [140, 239, 34]
    """
    import random
    if seed is not None:
        random.seed(seed)
    return [random.randint(0, 255) for _ in range(n_channels)]