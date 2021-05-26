import cv2

from utils import subplot, save_subplot
from utils import timing
from utils import replace_filepath_components
import config
from detect_icon.detect_icon import IconDetector


def detect_icon_func(img_path):
    img = cv2.imread(img_path) if isinstance(img_path, str) else img_path

    icon_detector = IconDetector()
    icon_boxes, door_boxes = icon_detector.predict_with_class(img)

    return icon_boxes, door_boxes



    if opt.remove_icon_version==1:
        img = remove_icon_1(org_img)
    else:
        raise ValueError('select opt.remove_icon_version in 1, 2')

    if opt.dev.remove_icon.save_result:
        save_path = replace_filepath_components(opt.img_path,
            new_contain_dir=config.REMOVE_ICON__RESULT__DIR, new_ext='.jpg')
        cv2.imwrite(save_path, img)

    if opt.dev.remove_icon.plot_compare_verions or opt.dev.remove_icon.save_compare_verions:
        img_v1 = img if opt.remove_icon_version==1 else remove_icon_1(org_img)

        imgs   = [org_img, img_v1]
        titles = ['Input image', 'Remove icon v1', 'Remove icon v2', 'Remove icon v3']

        if opt.dev.remove_icon.save_compare_verions:
            save_path = replace_filepath_components(opt.img_path,
                new_contain_dir=config.REMOVE_ICON__VERSIONS__DIR, new_ext='.jpg')
            save_subplot(imgs, titles, save_path=save_path, cols=len(imgs))

        if opt.dev.remove_icon.plot_compare_verions:
            subplot(imgs,titles, cols=len(imgs))

    if opt.dev.remove_icon.save_bna:
        save_path = replace_filepath_components(opt.img_path,
            new_contain_dir=config.REMOVE_ICON__BNA__DIR, new_ext='.jpg')
        save_subplot([org_img, img],
                     ['Before remove icons', 'After remove icons'],
                     save_path=save_path)

    if opt.dev.remove_icon.plot_bna:
        subplot([org_img, img],['Before remove icons', 'After remove icons'])

    return img