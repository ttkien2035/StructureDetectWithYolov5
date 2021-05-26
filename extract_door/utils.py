import inspect
import os
import re
import pickle
import time
import functools

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances


def img_from_box(img, box):
    x1,y1,x2,y2 = box[:4]
    return img[int(y1):int(y2), int(x1):int(x2)]


def calc_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_2way_iou( box1 , box2 ):
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


def get_boxes_inside_large_box(large_boxes, small_boxes):

    def to_skip(box, done_boxes=None):
        invalid_size = box[0] >= box[2] or box[1] >= box[3]
        done = tuple(box) in done_boxes if done_boxes is not None else False
        return invalid_size or done

    done_boxes = []
    result_boxes = []
    for large_box in large_boxes:
        small_boxes_in_large_box = []
        if to_skip(large_box):
            continue

        for small_box in small_boxes:
            if to_skip(small_box, done_boxes):
                continue
            _, _, r1, r2 = calculate_2way_iou(small_box, large_box)

            if r1 >= 0.5 or r2 >= 0.7:
                small_boxes_in_large_box.append(small_box)
                done_boxes.append(tuple(small_box))

        if not small_boxes_in_large_box:
            result_boxes.append(None)
            continue

        small_boxes_in_large_box = sorted(small_boxes_in_large_box, key=lambda x: x[0])
        result_boxes.append(small_boxes_in_large_box)
    return result_boxes


def debugPrint(variable, back_level = 1, name_only = False, direct_print=True):
    # start jump back to outer frame(s)
    frame = inspect.currentframe()
    for _ in range(back_level):
        frame = frame.f_back
    # get the name of the function and its params
    s = inspect.getframeinfo(frame).code_context[0]
    # filter get params
    var_name = re.search(r"\((.*)\)", s).group(1)
    # only get the name of param "variable", ignore "back_level"
    var_name = var_name.split(',')[0]
    if name_only:
        result_str = f"{var_name}"
    else:
        result_str = f"{var_name} = {variable}"

    if direct_print:
        print(result_str)
        return None

    return result_str


def _get_plt_cmap(cmap, img):
    if img.ndim == 2:
        return 'gray'

    elif cmap=='cv2':
        return 'bgr'

    elif cmap == 'rgb' or cmap is None:
        return None

    else:
        return cmap


def _imshow(img, title='debug', figsize='auto', cmap='cv2', show=True,
           save_path='not_passed'):
    """multi-type plotting function using matplotlib"""

    if save_path == 'not_passed': # use in save_imshow
        raise TypeError('save_path argument must be passed')

    if isinstance(img, str):
        if os.path.exists(img):
            img = cv2.imread(img)
        else:
            statement = 'Argument "img" should be a numpy array image or an '
            statement+= 'existing path to an image'
            raise ValueError(statement)

    if title == 'debug':
        # print img variable name:
        title = debugPrint(img, back_level=2,name_only=True, direct_print=False)

    if figsize == 'auto':
        # get height, width in pixel unit
        h,w,*_ = img.shape
        # convert pixel to inch
        matplotlib_dpi = 80
        w_inch = w / matplotlib_dpi
        h_inch = h / matplotlib_dpi
        # roundup
        w_inch = int(w_inch) + (w_inch % 1 > 0)
        h_inch = int(h_inch) + (h_inch % 1 > 0)
        # set figure size
        plt.figure(figsize=(w_inch, h_inch))
    elif figsize != None:
        plt.figure(figsize=(figsize))

    # if cmap == cv2, auto use gray scale or bgr mode
    if cmap == 'cv2':
        if img.ndim == 2:
            cmap = 'gray'
        elif img.ndim == 3:
            cmap = 'bgr'

    if cmap == "bgr":
        im = img.copy()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im)
    else:
        plt.imshow(img, cmap = cmap)

    if title is not None:
        plt.title(title)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


imshow      = functools.partial(_imshow, show=True, save_path=None)
save_imshow = functools.partial(_imshow, show=False, save_path='not_passed')


def _subplot(img_list, titles=None, figsize='auto', cols = 3,
            cmap = 'rgb', show=True, save_path=None):
    if save_path == 'not_passed': # use in save_imshow
        raise TypeError('save_path argument must be passed')

    cols = min([len(img_list), cols])

    n_img = len(img_list)
    rows = n_img // cols + 1

    # auto define figsize
    if figsize == 'auto':
        w = 8 * cols
        h = 14 * rows
        figsize = (w,h)
    plt.figure(figsize=figsize)

    # start subplot
    for ix, img in enumerate(img_list):
        ax = plt.subplot(rows, cols, ix+1)
        if titles is not None:
            ax.set_title(titles[ix],fontdict={'fontsize':22}) # tune this
        plt.xticks([]), plt.yticks([])
        cmap = _get_plt_cmap(cmap, img)
        if cmap == "bgr":
            img = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        else:
            ax.imshow(img, cmap = cmap)

    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


subplot      = functools.partial(_subplot, show=True, save_path=None)
save_subplot = functools.partial(_subplot, show=False, save_path='not_passed')


def get_dominant_color(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return tuple(colors[count.argmax()].tolist())


def fill_small_img(big_img, box, small_img, inplace=False):
    if not inplace:
        big_img = big_img.copy()
    x1,y1,x2,y2 = box
    big_img[y1:y2,x1:x2] = small_img
    return big_img


def walk_path(path,exts='img'):

    if exts == 'img':
        exts = ['jpg','jpeg','png']

    img_paths = [os.path.join(root, file)
                for root, dirs, files in os.walk(path)
                for file in files
                if  file.split('.')[-1].lower() in exts]
    return img_paths


def dump_pickle(obj, pickle_file_path):
    """
    example:
    dump_boxes_and_texts(my_list, 'test.pkl')
    """
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(obj,f)


def load_pickle(pickle_file_path):
    """
    example:
    my_list = load_pickle('test.pkl')
    """
    with open(pickle_file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def replace_ext(path:str or list, original_ext:list, new_ext:str) -> str or list:
    """replace original extension(s) with a new extension in path(s)"""
    assert isinstance(new_ext,str), "new_ext must be a string"

    if not isinstance(path,list):
        path = [path]
        return_str = True
    else:
        return_str = False

    if not isinstance(original_ext,list):
        original_ext = [original_ext]

    new_paths = []
    for p in path:
        splitted_path = p.split('.')
        if splitted_path[-1] in original_ext:
            splitted_path[-1] = new_ext
        new_paths.append('.'.join(splitted_path))

    if return_str:
        new_paths = new_paths[0]

    return new_paths


def split_filepath(filepath):
    from pathlib import Path

    path        = Path(filepath).resolve()
    contain_dir = str(path.parent)
    filename    = path.stem
    ext         = path.suffix

    return contain_dir, filename, ext


def join_filepath_components(contain_dir:str, filename:str, ext:str):
    if len(ext)!=0 and ext[0] != '.':
        ext = '.' + ext
    return os.path.join(contain_dir, filename+ext)


def resolve_path(path:str)->str:
    from pathlib import Path
    return str(Path(path).resolve())


def replace_filepath_components(filepath:str, new_contain_dir:str=None,
                                new_filename:str=None, new_ext:str=None,
                                resolve=False)->str:
    from pathlib import Path

    # if all are None
    if not(any([new_contain_dir, new_filename, new_ext])):
        return str(Path(filepath).resolve())
    contain_dir, filename, ext = split_filepath(filepath)
    new_contain_dir = contain_dir if new_contain_dir is None else new_contain_dir
    new_filename    = filename    if new_filename    is None else new_filename
    new_ext         = ext         if new_ext         is None else new_ext
    new_filepath    = join_filepath_components(new_contain_dir, new_filename, new_ext)
    if resolve:
        new_filepath = resolve_path(new_filepath)
    return new_filepath


def timing(func=None, *, activate=True, return_time=False, factor=1000,
           split_before=False, split_after=False):
    def decor_timing(func):
        time_run = 0
        @functools.wraps(func)
        def wrap_func(*args,**kwargs):
            nonlocal time_run

            time1 = time.time()
            ret = func(*args,**kwargs)
            time2 = time.time()
            run_time = time2 - time1

            if split_before:
                print(f"{' '+str(time_run)+' ':#^56}")
                time_run +=1

            print('Runtime of {:<30s}{:>12.5f} ms'.format(
                func.__name__, run_time*factor))

            if split_after:
                print(f"{' '+str(time_run)+' ':#^56}")
                time_run +=1

            if return_time:
                return run_time*factor
            else:
                return ret
        if activate:
            return wrap_func
        else:
            return func

    if func is None:
        return decor_timing
    else:
        return decor_timing(func)


def is_env_var_equal(varible, value):
    return os.environ.get(varible) == value


def do_timer():
    return is_env_var_equal('LOG_TIME','True')


def get_attrs(cls):
    return [i for i in cls.__dict__.keys() if i[:1] != '_']


def get_arr_frequency(arr, ascending=True, to_ratio=False)->dict: # -> freq_dict

    unique, counts = np.unique(arr, return_counts=True)

    sorted_ix = np.argsort(counts)
    if not ascending:
        sorted_ix = sorted_ix[::-1]

    unique = unique[sorted_ix]
    counts = counts[sorted_ix]

    if to_ratio:
        counts = list(map(lambda x: x/sum(counts), counts))

    return {k:v for k,v in zip(unique, counts)}


def hconcat_images(imgs):
    if len(imgs)==0:
        return None
    output_h = max(img.shape[0] for img in imgs)
    padded_imgs = []
    for img in imgs:
        pad_size = output_h - img.shape[0]
        padded_img = cv2.copyMakeBorder(
            img, pad_size//2, pad_size - pad_size //2, 0, 0,
            cv2.BORDER_CONSTANT, value = (255, 255, 255))
        padded_imgs.append(padded_img)
    output_img = np.hstack(tuple(padded_imgs))
    return output_img


def resize_match(src_img, target_img):
    h,w,*_ = target_img.shape
    return cv2.resize(src_img,(w,h))


class SingletonMeta(type):
    """
    example:
    class PlaceCorrection(metaclass=SingletonMeta):
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def make_box_of_color(color):
    """
    CAUTION: notice your cmap
    This function create a numpy array image of the color. It does not include drawing
    example: make_box_of_color((51, 230, 253))
    """
    h, w = (10,10)
    img = np.repeat([color], repeats = w, axis=0).astype('uint8')
    img = np.repeat([img],   repeats = h, axis=0)
    return img