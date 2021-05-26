from detect_icon_func import detect_icon_func
import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    img_paths = glob.glob('/home/tducnguyen/NguyenTran/Project/30_Architecture_plan_clean/Code/Architecture_Plan_Cleaner/Clone_P/plan_cleaner/results/crop/result/*.jpg')
    for img_id in tqdm(range(len(img_paths))):
        img_path = img_paths[img_id]
        img = cv2.imread(img_path)
        img_name = img_path.split('/')[-1].split('.')[0]
        boxes_icon, boxes_door = detect_icon_func(img)
        count = 1
        for box in boxes_door:
            img_door = img[box[1]: box[3], box[0]: box[2], :]
            try:
                cv2.imwrite('box_door/' + img_name + '_{}.jpg'.format(count), img_door)
                count += 1
            except:
                print()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

