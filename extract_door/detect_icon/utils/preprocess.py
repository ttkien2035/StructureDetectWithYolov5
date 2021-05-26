import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def create_table_struct(img):
    """
    This function will create a simple table frame of the image
    """
    binary_img = img.copy()
    # Invert the image
    binary_img = ~binary_img
    scale = 25 # bigger scale allow keeping more lines
    vertical_length = np.array(img).shape[0]//scale
    horizontal_length = np.array(img).shape[1]//scale

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    # Morphological operation to detect vertical lines from an image
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length))
    img_temp1 = cv2.erode(binary_img, vertical_kernel, iterations=1) # remove noise
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=2) # connect short lines
    vertical_lines_img = cv2.erode(vertical_lines_img, vertical_kernel, iterations=1) # reduce line length

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    # Morphological operation to detect horizontal lines from an image
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1))
    img_temp2 = cv2.erode(binary_img, horizontal_kernel, iterations=1) # remove noise
    horizontal_lines_img = cv2.dilate(img_temp2, horizontal_kernel, iterations=2) # connect short lines
    horizontal_lines_img = cv2.erode(horizontal_lines_img, horizontal_kernel, iterations=1) # reduce line length

    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    alpha = 0.5
    beta = 1.0 - alpha
    final_binary_img = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
#     final_binary_img = cv2.dilate(final_binary_img, kernel, iterations=1)
    (thresh, final_binary_img) = cv2.threshold(final_binary_img, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return final_binary_img


def get_table_mask(org_img):
    # org_img = cv2.imread(img_path)
    img = org_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = img.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    processed_img = clahe.apply(processed_img)
    processed_img = cv2.adaptiveThreshold(processed_img, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY , 7, 4)

    line_mask = create_table_struct(processed_img)
    return line_mask

def resize_canvas(image, canvasSize, color = (255,255,255)): 
    """
    >>> Input
        -> image: np.array
        -> cavasSize: tuple-(1,2)
        -> color: tuple-(1,3)
    >>> Return value: np.array
    """
    bottom = max(0, canvasSize[0] - image.shape[0])
    right = max(0, canvasSize[1] - image.shape[1])
    resultImg = cv2.copyMakeBorder(image, top = 0, bottom = bottom, left = 0, right = right, borderType = cv2.BORDER_CONSTANT, value = color)
    return resultImg

def separate_image(image:np.ndarray, tileSize:tuple, overlap:int = 0,
    debug = False):
    """
    >>> Input
        -> image: np.array
        -> tileSize: tuple-(1,2)
        -> overlap: int
        -> debug: boolean
    >>> Return value: np.array images and np.array boxes

    example:
    
    """
    import math
    if isinstance(tileSize,int):
        tileSize = (tileSize,tileSize)
    h, w = image.shape[:2]
    dist = (tileSize[0]-overlap, tileSize[1]-overlap)
    h_resized = math.ceil((h - overlap)/dist[0])*dist[0]+overlap
    w_resized = math.ceil((w - overlap)/dist[1])*dist[1]+overlap
    resized = resize_canvas(image, (h_resized, w_resized))

    x = np.arange(0, w_resized-tileSize[1]+1, dist[1])
    y = np.arange(0, h_resized-tileSize[0]+1, dist[0])
    boxes = []
    result = []
    for i in y:
        box_temp = []
        result_temp = []
        for j in x:
            xmin,ymin,xmax,ymax = j,i,j+tileSize[1],i+tileSize[0]
            box_temp.append([xmin,ymin,xmax,ymax])
            result_temp.append(resized[ymin:ymax, xmin:xmax])
        boxes.append(box_temp)
        result.append(result_temp)

    if debug:
        fig, ax = plt.subplots(y.shape[0], x.shape[0],figsize=(20,20))
        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                ax[i,j].imshow(result[i][j])
        plt.show()
        plt.close(fig)
    return np.array(result), np.array(boxes)

def preprocess(image, colorSpace = None, imgSize = None):
    """ 
    >>> Input
        -> image: np.array
        -> colorSpace: str (e.g., 'rgb', 'gray')
        -> color: tuple-(1,3)
    >>> Return value: np.array, np.array
    """
    assert isinstance(image, np.ndarray) or isinstance(image, str),\
    "'img' must be a string or an numpy array."
    
    if isinstance(image, str): 
        oriImg = cv2.imread(image)
    else: 
        oriImg =  image
    
    if oriImg is None: return None, None
    
    resultImg = oriImg.copy()
    
    if imgSize:
        try:
            resultImg = cv2.resize(resultImg, imgSize)
        except Exception as e:
            print(e)
        
    if colorSpace:
        try:
            cvtArg = eval(f'cv2.COLOR_BGR2{colorSpace.upper()}')
            resultImg = cv2.cvtColor(resultImg, cvtArg)
        except Exception as e:
            print(e)
        
    return resultImg, oriImg